"""
データ読み込み・前処理モジュール

boxscoreとgamesデータを結合し、分析用のデータフレームを作成する
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# 各スタッツの計測開始シーズン（seasonStartYear）
# このシーズンより前のデータはNaNに変換される
STAT_START_SEASONS = {
    "TRB": 1950,   # 1950-51シーズンから
    "STL": 1973,   # 1973-74シーズンから
    "BLK": 1973,   # 1973-74シーズンから
    "ORB": 1973,   # 1973-74シーズンから
    "DRB": 1973,   # 1973-74シーズンから
    "TOV": 1977,   # 1977-78シーズンから
    "3P": 1979,    # 1979-80シーズンから
    "3PA": 1979,   # 1979-80シーズンから
    "+/-": 1996,   # 1996-97シーズンから
}


class NBADataLoader:
    """NBAデータの読み込みと前処理を行うクラス"""

    def __init__(self, data_dir: str = "data"):
        """
        Parameters
        ----------
        data_dir : str
            データディレクトリのパス
        """
        self.data_dir = Path(data_dir)
        self._boxscore: Optional[pd.DataFrame] = None
        self._games: Optional[pd.DataFrame] = None
        self._player_info: Optional[pd.DataFrame] = None
        self._merged_df: Optional[pd.DataFrame] = None

    # =========================================================================
    # データ読み込み
    # =========================================================================

    def load_boxscore(self, filename: str = "boxscore1946-2025.csv.gz") -> pd.DataFrame:
        """boxscoreデータを読み込む（gzip圧縮対応）

        pandasは.gzファイルを自動的に認識して解凍します。
        """
        filepath = self.data_dir / filename
        self._boxscore = pd.read_csv(filepath)
        return self._boxscore

    def load_games(self, filename: str = "games1946-2025.csv.gz") -> pd.DataFrame:
        """gamesデータを読み込む（gzip圧縮対応）

        pandasは.gzファイルを自動的に認識して解凍します。
        """
        filepath = self.data_dir / filename
        self._games = pd.read_csv(filepath)
        return self._games

    def load_player_info(self, filename: str = "Players_data_Latest.csv") -> pd.DataFrame:
        """選手情報データを読み込む"""
        filepath = self.data_dir / filename
        self._player_info = pd.read_csv(filepath)
        if "birth_date" in self._player_info.columns:
            self._player_info["birth_date"] = pd.to_datetime(
                self._player_info["birth_date"], errors="coerce"
            )
        return self._player_info

    # =========================================================================
    # データ前処理・クリーニング
    # =========================================================================

    def create_analysis_df(
        self,
        boxscore: Optional[pd.DataFrame] = None,
        games: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        boxscoreとgamesを結合し、分析用の派生列を追加したデータフレームを作成

        Parameters
        ----------
        boxscore : pd.DataFrame, optional
            boxscoreデータ（指定しない場合は事前に読み込んだデータを使用）
        games : pd.DataFrame, optional
            gamesデータ（指定しない場合は事前に読み込んだデータを使用）

        Returns
        -------
        pd.DataFrame
            分析用データフレーム
        """
        bs = boxscore if boxscore is not None else self._boxscore
        gm = games if games is not None else self._games

        if bs is None or gm is None:
            raise ValueError("boxscoreとgamesデータを先に読み込んでください")

        # gamesとマージ
        merge_cols = [
            "seasonStartYear", "League", "isRegular", "isFinal", "isPlayin",
            "Winner", "game_id", "Arena", "datetime"
        ]
        available_cols = [c for c in merge_cols if c in gm.columns]
        df = bs.merge(gm[available_cols], on="game_id")

        # 計測開始前のスタッツをNaNに変換
        df = self._nullify_pre_tracking_stats(df)

        # 派生列を追加
        df = self._add_derived_columns(df)

        # 日時でソートしてリセット
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime").reset_index(drop=True)

        # シーズン列を追加
        df["season"] = df["seasonStartYear"].apply(
            lambda x: f"{int(x)}-{int(x)+1}" if pd.notna(x) else None
        )

        self._merged_df = df
        return df

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """派生列（統計フラグなど）を追加"""
        # 勝敗フラグ
        df["Win"] = np.where(df["teamName"] == df["Winner"], 1, 0)
        df["Lose"] = np.where(df["Win"] == 1, 0, 1)

        # 出場フラグ（スタッツが1つでもあるか）
        stat_cols = ["FG", "FGA", "3P", "3PA", "FT", "FTA", "ORB", "DRB",
                     "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "+/-"]
        df["Played"] = np.where(
            (df[stat_cols] != 0).any(axis=1) | (df["MP"] != "0"), 1, 0
        )

        # ダブルダブル・トリプルダブル判定
        df = self._add_double_flags(df)

        # 2Pシュート
        df["2P"] = df["FG"] - df["3P"]
        df["2PA"] = df["FGA"] - df["3PA"]
        df["Stocks"] = df["STL"] + df["BLK"]

        # 各種スタッツ閾値フラグ
        df = self._add_stat_threshold_flags(df)

        # TOV関連
        df["TOV_0"] = np.where(df["TOV"] == 0, 1, 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ast_tov_ratio = df["AST"] / df["TOV"]
            df["ASTTOV>=3"] = np.where(ast_tov_ratio >= 3, 1, 0)

        return df

    def _add_double_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """ダブルダブル・トリプルダブルなどのフラグを追加"""
        pts10 = (df["PTS"] >= 10).astype(int)
        ast10 = (df["AST"] >= 10).astype(int)
        trb10 = (df["TRB"] >= 10).astype(int)
        stl10 = (df["STL"] >= 10).astype(int)
        blk10 = (df["BLK"] >= 10).astype(int)

        doubles_count = pts10 + ast10 + trb10 + stl10 + blk10

        df["DD"] = np.where(doubles_count == 2, 1, 0)  # ダブルダブル
        df["TD"] = np.where(doubles_count == 3, 1, 0)  # トリプルダブル
        df["QD"] = np.where(doubles_count == 4, 1, 0)  # クアドラプルダブル

        return df

    def _add_stat_threshold_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """各種スタッツの閾値フラグを追加"""
        # 得点
        for pts in [10, 20, 25, 30, 40, 50]:
            df[f"{pts}PTS+"] = np.where(df["PTS"] >= pts, 1, 0)

        # オフェンスリバウンド
        for orb in [5, 10]:
            df[f"{orb}ORB+"] = np.where(df["ORB"] >= orb, 1, 0)

        # トータルリバウンド
        for trb in [10, 15, 20, 25, 30]:
            df[f"{trb}TRB+"] = np.where(df["TRB"] >= trb, 1, 0)

        # アシスト
        for ast in [10, 15, 20, 25]:
            df[f"{ast}AST+"] = np.where(df["AST"] >= ast, 1, 0)

        # 3ポイント
        df["5_3P+"] = np.where(df["3P"] >= 5, 1, 0)
        df["3P_1+"] = np.where(df["3P"] >= 1, 1, 0)

        # 複合ダブルダブル
        df["AST&PTS_DD"] = np.where(
            (df["AST"] >= 10) & (df["PTS"] >= 10), 1, 0
        )
        df["TRB&PTS_DD"] = np.where(
            (df["TRB"] >= 10) & (df["PTS"] >= 10), 1, 0
        )
        df["20PTS_20TRB"] = np.where(
            (df["PTS"] >= 20) & (df["TRB"] >= 20), 1, 0
        )

        return df

    def _nullify_pre_tracking_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計測開始前のスタッツをNaNに変換

        各スタッツは計測開始時期が異なるため、計測開始前のデータを
        NaNに変換して誤った分析を防ぐ
        """
        for stat, start_year in STAT_START_SEASONS.items():
            if stat in df.columns:
                mask = df["seasonStartYear"] < start_year
                df.loc[mask, stat] = np.nan
        return df

    # =========================================================================
    # 年齢関連の処理
    # =========================================================================

    def add_age_columns(
        self,
        df: pd.DataFrame,
        player_info: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        選手の年齢関連列を追加

        Parameters
        ----------
        df : pd.DataFrame
            分析用データフレーム
        player_info : pd.DataFrame, optional
            選手情報データ

        Returns
        -------
        pd.DataFrame
            年齢列を追加したデータフレーム
        """
        pi = player_info if player_info is not None else self._player_info

        if pi is None:
            raise ValueError("player_infoデータを先に読み込んでください")

        # 選手情報をマージ
        df = df.merge(pi, left_on="playerName", right_on="name", how="left")

        # シーズン終了年
        df["season_end"] = df["season"].str.split("-").str[1].astype(float)

        # 生年月日から年齢計算
        df["birth_year"] = df["birth_date"].dt.year
        df["birth_month"] = df["birth_date"].dt.month
        df["birth_day"] = df["birth_date"].dt.day

        # 30歳到達日
        df["birth_date_30years"] = df["birth_date"] + pd.DateOffset(years=30)
        df["is_30years_old"] = np.where(
            df["datetime"] >= df["birth_date_30years"], 1, 0
        )

        # 単純年齢
        df["age"] = df["season_end"] - df["birth_year"]

        # シーズン終了時の年齢（月日を考慮）
        regular_season = df[df["isRegular"] == 1]
        season_end_day = regular_season.groupby("season")["datetime"].transform("max")
        df["season_end_day"] = season_end_day

        df["age_at_season_end"] = np.where(
            df["birth_month"] > df["season_end_day"].dt.month,
            df["season_end"] - df["birth_year"] - 1,
            df["season_end"] - df["birth_year"]
        )

        # 試合日時点での正確な年齢（誕生日を考慮）
        # 年の差を計算し、まだ誕生日を迎えていない場合は1を引く
        df["age_at_game"] = (
            df["datetime"].dt.year - df["birth_date"].dt.year -
            (
                (df["datetime"].dt.month < df["birth_date"].dt.month) |
                (
                    (df["datetime"].dt.month == df["birth_date"].dt.month) &
                    (df["datetime"].dt.day < df["birth_date"].dt.day)
                )
            ).astype(int)
        )

        return df

    # =========================================================================
    # プロパティ
    # =========================================================================

    @property
    def boxscore(self) -> Optional[pd.DataFrame]:
        return self._boxscore

    @property
    def games(self) -> Optional[pd.DataFrame]:
        return self._games

    @property
    def player_info(self) -> Optional[pd.DataFrame]:
        return self._player_info

    @property
    def merged_df(self) -> Optional[pd.DataFrame]:
        return self._merged_df

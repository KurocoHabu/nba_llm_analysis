"""
データ読み込み・前処理モジュール（Polars版）

boxscoreとgamesデータを結合し、分析用のデータフレームを作成する

Databricks Apps環境では、Databricks SDKを経由してVolumeにアクセスする
"""

import os
import io
import gzip
import polars as pl
from pathlib import Path
from typing import Optional


# 各スタッツの計測開始シーズン（seasonStartYear）
# このシーズンより前のデータはNullに変換される
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


def is_databricks_apps(data_dir: str = None) -> bool:
    """Databricks Apps環境かどうかを判定

    判定方法:
    1. 環境変数 DATABRICKS_APPS が "true" の場合
    2. DATA_DIR が /Volumes で始まる場合
    """
    # 環境変数で明示的に指定されている場合
    if os.environ.get("DATABRICKS_APPS") == "true":
        print("[DEBUG] is_databricks_apps: True (via DATABRICKS_APPS env)")
        return True

    # DATA_DIRがVolumeパスの場合
    data_dir_env = data_dir or os.environ.get("DATA_DIR", "")
    if data_dir_env.startswith("/Volumes"):
        print(f"[DEBUG] is_databricks_apps: True (DATA_DIR={data_dir_env})")
        return True

    print(f"[DEBUG] is_databricks_apps: False (DATABRICKS_APPS={os.environ.get('DATABRICKS_APPS')}, DATA_DIR={data_dir_env})")
    return False


def download_from_volume(volume_path: str) -> bytes:
    """
    Databricks SDKを使ってVolumeからファイルをダウンロード

    Parameters
    ----------
    volume_path : str
        Volumeのパス（例: /Volumes/workspace/nba_analysis/data/file.csv）

    Returns
    -------
    bytes
        ファイルの内容
    """
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    response = w.files.download(volume_path)
    return response.contents.read()


class NBADataLoader:
    """NBAデータの読み込みと前処理を行うクラス（Polars版）"""

    def __init__(self, data_dir: str = "data"):
        """
        Parameters
        ----------
        data_dir : str
            データディレクトリのパス
            Databricks Apps環境では、Volumeのパス（例: /Volumes/workspace/nba_analysis/data）
        """
        self.data_dir = Path(data_dir)
        self._is_databricks = is_databricks_apps(data_dir)
        print(f"[DEBUG] NBADataLoader initialized: data_dir={data_dir}, is_databricks={self._is_databricks}")
        self._boxscore: Optional[pl.DataFrame] = None
        self._games: Optional[pl.DataFrame] = None
        self._player_info: Optional[pl.DataFrame] = None
        self._merged_df: Optional[pl.DataFrame] = None

    # =========================================================================
    # データ読み込み
    # =========================================================================

    def _read_csv_file(self, filename: str, schema_overrides: Optional[dict] = None) -> pl.DataFrame:
        """
        CSVファイルを読み込む（環境に応じて適切な方法を選択）

        Parameters
        ----------
        filename : str
            ファイル名
        schema_overrides : dict, optional
            スキーマオーバーライド

        Returns
        -------
        pl.DataFrame
            読み込んだデータフレーム
        """
        filepath = str(self.data_dir / filename)
        print(f"[DEBUG] _read_csv_file: filepath={filepath}, is_databricks={self._is_databricks}")

        if self._is_databricks:
            # Databricks Apps: SDKでダウンロードしてからPolarsで読み込む
            file_content = download_from_volume(filepath)

            # gzip圧縮ファイルの場合は解凍
            if filename.endswith('.gz'):
                file_content = gzip.decompress(file_content)

            # バイトデータからPolarsで読み込み
            try:
                # Polars >= 0.19
                return pl.read_csv(
                    io.BytesIO(file_content),
                    schema_overrides=schema_overrides,
                    infer_schema_length=10000,
                )
            except TypeError:
                # Polars < 0.19
                return pl.read_csv(
                    io.BytesIO(file_content),
                    dtypes=schema_overrides,
                    infer_schema_length=10000,
                )
        else:
            # ローカル/Streamlit Cloud: 直接ファイルパスから読み込み
            try:
                # Polars >= 0.19
                return pl.read_csv(
                    filepath,
                    schema_overrides=schema_overrides,
                    infer_schema_length=10000,
                )
            except TypeError:
                # Polars < 0.19
                return pl.read_csv(
                    filepath,
                    dtypes=schema_overrides,
                    infer_schema_length=10000,
                )

    def load_boxscore(self, filename: str = "boxscore1946-2025.csv.gz") -> pl.DataFrame:
        """boxscoreデータを読み込む（gzip圧縮対応）"""
        # MP列は時間形式（"48:00"など）なので文字列として読み込む
        self._boxscore = self._read_csv_file(filename, schema_overrides={"MP": pl.Utf8})
        return self._boxscore

    def load_games(self, filename: str = "games1946-2025.csv.gz") -> pl.DataFrame:
        """gamesデータを読み込む（gzip圧縮対応）"""
        self._games = self._read_csv_file(filename)
        return self._games

    def load_player_info(self, filename: str = "Players_data_Latest.csv") -> pl.DataFrame:
        """選手情報データを読み込む"""
        self._player_info = self._read_csv_file(filename)
        if "birth_date" in self._player_info.columns:
            self._player_info = self._player_info.with_columns(
                pl.col("birth_date").str.to_datetime(format="%Y-%m-%d", strict=False)
            )
        return self._player_info

    # =========================================================================
    # データ前処理・クリーニング
    # =========================================================================

    def create_analysis_df(
        self,
        boxscore: Optional[pl.DataFrame] = None,
        games: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """
        boxscoreとgamesを結合し、分析用の派生列を追加したデータフレームを作成

        Parameters
        ----------
        boxscore : pl.DataFrame, optional
            boxscoreデータ（指定しない場合は事前に読み込んだデータを使用）
        games : pl.DataFrame, optional
            gamesデータ（指定しない場合は事前に読み込んだデータを使用）

        Returns
        -------
        pl.DataFrame
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
        df = bs.join(gm.select(available_cols), on="game_id")

        # 計測開始前のスタッツをNullに変換
        df = self._nullify_pre_tracking_stats(df)

        # 派生列を追加
        df = self._add_derived_columns(df)

        # 日時でソート
        df = df.with_columns(
            pl.col("datetime").str.to_datetime(strict=False)
        ).sort("datetime")

        # シーズン列を追加
        df = df.with_columns(
            pl.when(pl.col("seasonStartYear").is_not_null())
            .then(
                pl.col("seasonStartYear").cast(pl.Int64).cast(pl.Utf8) + "-" +
                (pl.col("seasonStartYear").cast(pl.Int64) + 1).cast(pl.Utf8)
            )
            .otherwise(None)
            .alias("season")
        )

        self._merged_df = df
        return df

    def _add_derived_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """派生列（統計フラグなど）を追加"""
        # 勝敗フラグ
        df = df.with_columns([
            pl.when(pl.col("teamName") == pl.col("Winner"))
            .then(1).otherwise(0).alias("Win"),
            pl.when(pl.col("teamName") == pl.col("Winner"))
            .then(0).otherwise(1).alias("Lose"),
        ])

        # 出場フラグ（スタッツが1つでもあるか）
        stat_cols = ["FG", "FGA", "3P", "3PA", "FT", "FTA", "ORB", "DRB",
                     "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "+/-"]
        available_stat_cols = [c for c in stat_cols if c in df.columns]

        # 各スタッツが0でないかチェック
        stat_check = pl.lit(False)
        for col in available_stat_cols:
            stat_check = stat_check | (pl.col(col) != 0)

        df = df.with_columns(
            pl.when(stat_check | (pl.col("MP") != "0"))
            .then(1).otherwise(0).alias("Played")
        )

        # ダブルダブル・トリプルダブル判定
        df = self._add_double_flags(df)

        # 2Pシュート
        df = df.with_columns([
            (pl.col("FG") - pl.col("3P")).alias("2P"),
            (pl.col("FGA") - pl.col("3PA")).alias("2PA"),
            (pl.col("STL") + pl.col("BLK")).alias("Stocks"),
        ])

        # 各種スタッツ閾値フラグ
        df = self._add_stat_threshold_flags(df)

        # TOV関連
        df = df.with_columns([
            pl.when(pl.col("TOV") == 0).then(1).otherwise(0).alias("TOV_0"),
            pl.when((pl.col("TOV") != 0) & (pl.col("AST") / pl.col("TOV") >= 3))
            .then(1).otherwise(0).alias("ASTTOV>=3"),
        ])

        return df

    def _add_double_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """ダブルダブル・トリプルダブルなどのフラグを追加"""
        df = df.with_columns([
            (pl.col("PTS") >= 10).cast(pl.Int64).alias("_pts10"),
            (pl.col("AST") >= 10).cast(pl.Int64).alias("_ast10"),
            (pl.col("TRB") >= 10).cast(pl.Int64).alias("_trb10"),
            (pl.col("STL") >= 10).cast(pl.Int64).alias("_stl10"),
            (pl.col("BLK") >= 10).cast(pl.Int64).alias("_blk10"),
        ])

        df = df.with_columns(
            (pl.col("_pts10") + pl.col("_ast10") + pl.col("_trb10") +
             pl.col("_stl10") + pl.col("_blk10")).alias("_doubles_count")
        )

        df = df.with_columns([
            pl.when(pl.col("_doubles_count") == 2).then(1).otherwise(0).alias("DD"),
            pl.when(pl.col("_doubles_count") == 3).then(1).otherwise(0).alias("TD"),
            pl.when(pl.col("_doubles_count") == 4).then(1).otherwise(0).alias("QD"),
        ])

        # 一時列を削除
        df = df.drop(["_pts10", "_ast10", "_trb10", "_stl10", "_blk10", "_doubles_count"])

        return df

    def _add_stat_threshold_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """各種スタッツの閾値フラグを追加"""
        # 得点
        for pts in [10, 20, 25, 30, 40, 50]:
            df = df.with_columns(
                pl.when(pl.col("PTS") >= pts).then(1).otherwise(0).alias(f"{pts}PTS+")
            )

        # オフェンスリバウンド
        for orb in [5, 10]:
            df = df.with_columns(
                pl.when(pl.col("ORB") >= orb).then(1).otherwise(0).alias(f"{orb}ORB+")
            )

        # トータルリバウンド
        for trb in [10, 15, 20, 25, 30]:
            df = df.with_columns(
                pl.when(pl.col("TRB") >= trb).then(1).otherwise(0).alias(f"{trb}TRB+")
            )

        # アシスト
        for ast in [10, 15, 20, 25]:
            df = df.with_columns(
                pl.when(pl.col("AST") >= ast).then(1).otherwise(0).alias(f"{ast}AST+")
            )

        # 3ポイント
        df = df.with_columns([
            pl.when(pl.col("3P") >= 5).then(1).otherwise(0).alias("5_3P+"),
            pl.when(pl.col("3P") >= 1).then(1).otherwise(0).alias("3P_1+"),
        ])

        # 複合ダブルダブル
        df = df.with_columns([
            pl.when((pl.col("AST") >= 10) & (pl.col("PTS") >= 10))
            .then(1).otherwise(0).alias("AST&PTS_DD"),
            pl.when((pl.col("TRB") >= 10) & (pl.col("PTS") >= 10))
            .then(1).otherwise(0).alias("TRB&PTS_DD"),
            pl.when((pl.col("PTS") >= 20) & (pl.col("TRB") >= 20))
            .then(1).otherwise(0).alias("20PTS_20TRB"),
        ])

        return df

    def _nullify_pre_tracking_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        計測開始前のスタッツをNullに変換

        各スタッツは計測開始時期が異なるため、計測開始前のデータを
        Nullに変換して誤った分析を防ぐ
        """
        for stat, start_year in STAT_START_SEASONS.items():
            if stat in df.columns:
                df = df.with_columns(
                    pl.when(pl.col("seasonStartYear") < start_year)
                    .then(None)
                    .otherwise(pl.col(stat))
                    .alias(stat)
                )
        return df

    # =========================================================================
    # 年齢関連の処理
    # =========================================================================

    def add_age_columns(
        self,
        df: pl.DataFrame,
        player_info: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """
        選手の年齢関連列を追加

        Parameters
        ----------
        df : pl.DataFrame
            分析用データフレーム
        player_info : pl.DataFrame, optional
            選手情報データ

        Returns
        -------
        pl.DataFrame
            年齢列を追加したデータフレーム
        """
        pi = player_info if player_info is not None else self._player_info

        if pi is None:
            raise ValueError("player_infoデータを先に読み込んでください")

        # 選手情報をマージ
        df = df.join(pi, left_on="playerName", right_on="name", how="left")

        # シーズン終了年
        df = df.with_columns(
            pl.col("season").str.split("-").list.get(1).cast(pl.Float64).alias("season_end")
        )

        # 生年月日から年齢計算
        df = df.with_columns([
            pl.col("birth_date").dt.year().alias("birth_year"),
            pl.col("birth_date").dt.month().alias("birth_month"),
            pl.col("birth_date").dt.day().alias("birth_day"),
        ])

        # 30歳到達日
        df = df.with_columns(
            (pl.col("birth_date") + pl.duration(days=365 * 30)).alias("birth_date_30years")
        )
        df = df.with_columns(
            pl.when(pl.col("datetime") >= pl.col("birth_date_30years"))
            .then(1).otherwise(0).alias("is_30years_old")
        )

        # 単純年齢
        df = df.with_columns(
            (pl.col("season_end") - pl.col("birth_year")).alias("age")
        )

        # シーズン終了時の年齢（月日を考慮）- 簡易版
        df = df.with_columns(
            pl.when(pl.col("birth_month") > 6)
            .then(pl.col("season_end") - pl.col("birth_year") - 1)
            .otherwise(pl.col("season_end") - pl.col("birth_year"))
            .alias("age_at_season_end")
        )

        # 試合日時点での正確な年齢（誕生日を考慮）
        df = df.with_columns(
            (
                pl.col("datetime").dt.year() - pl.col("birth_date").dt.year() -
                (
                    (pl.col("datetime").dt.month() < pl.col("birth_date").dt.month()) |
                    (
                        (pl.col("datetime").dt.month() == pl.col("birth_date").dt.month()) &
                        (pl.col("datetime").dt.day() < pl.col("birth_date").dt.day())
                    )
                ).cast(pl.Int64)
            ).alias("age_at_game")
        )

        return df

    # =========================================================================
    # プロパティ
    # =========================================================================

    @property
    def boxscore(self) -> Optional[pl.DataFrame]:
        return self._boxscore

    @property
    def games(self) -> Optional[pl.DataFrame]:
        return self._games

    @property
    def player_info(self) -> Optional[pl.DataFrame]:
        return self._player_info

    @property
    def merged_df(self) -> Optional[pl.DataFrame]:
        return self._merged_df

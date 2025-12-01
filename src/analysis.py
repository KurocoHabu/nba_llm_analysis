"""
NBA統計分析モジュール

各種分析関数を提供:
- 連続試合記録（〇〇得点連続試合など）
- 到達試合数（累計〇〇ポイント到達に必要な試合数）
- 年間達成回数（年間〇〇得点達成回数）
- 年齢別記録（30歳までの記録など）
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


# 同姓同名の選手（複数人が存在するため分析から除外）
DUPLICATE_NAME_PLAYERS = [
    "Eddie Johnson",
    "George Johnson",
    "Mike Dunleavy",
    "David Lee",
    "Jim Paxson",
    "Larry Johnson",
    "Matt Guokas",
]


class NBAAnalyzer:
    """NBA統計データの分析を行うクラス"""

    def __init__(
        self,
        df: pd.DataFrame,
        exclude_duplicate_names: bool = True,
        exclude_players: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame
            NBADataLoader.create_analysis_df()で作成した分析用データフレーム
        exclude_duplicate_names : bool
            同姓同名の選手を自動除外するか（デフォルト: True）
        exclude_players : List[str], optional
            追加で除外する選手名リスト
        """
        self._original_df = df

        # 除外リストを作成
        self._exclude_players = []
        if exclude_duplicate_names:
            self._exclude_players.extend(DUPLICATE_NAME_PLAYERS)
        if exclude_players:
            self._exclude_players.extend(exclude_players)

        # 除外を適用
        if self._exclude_players:
            self.df = df[~df["playerName"].isin(self._exclude_players)].copy()
            excluded_count = len(df) - len(self.df)
            if excluded_count > 0:
                print(f"※ 同姓同名選手 {len(self._exclude_players)}名 を除外しました（{excluded_count:,}行）")
        else:
            self.df = df

    # =========================================================================
    # 1. 連続試合記録の分析
    # =========================================================================

    def get_consecutive_games(
        self,
        label: str,
        game_type: Literal["regular", "playoff", "final", "all"] = "regular",
        league: str = "NBA",
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        連続試合記録（例: 20得点以上連続試合数）のランキングを取得

        Parameters
        ----------
        label : str
            対象の列名（例: "20PTS+", "10AST+", "DD"）
        game_type : str
            "regular": レギュラーシーズンのみ
            "playoff": プレイオフのみ
            "all": 全試合
        league : str
            リーグ（デフォルト: "NBA"）
        top_n : int
            上位N件を取得

        Returns
        -------
        pd.DataFrame
            選手ごとの最大連続試合数
        """
        # データフィルタリング
        data = self._filter_by_game_type(game_type, league)
        data = data[data["Played"] == 1]

        # 選手ごとに連続記録を計算
        result = data.groupby("playerName").apply(
            lambda x: self._count_max_consecutive(x[label].values),
            include_groups=False
        ).reset_index()
        result.columns = ["playerName", label]

        # ソートして上位を取得
        result = result.sort_values(label, ascending=False).head(top_n)

        return result

    @staticmethod
    def _count_max_consecutive(values: np.ndarray) -> int:
        """1が連続する最大回数をカウント"""
        count = 0
        max_count = 0
        for val in values:
            if val == 1:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
        return max_count

    def get_multiple_consecutive_games(
        self,
        labels: List[str],
        game_type: Literal["regular", "playoff", "final", "all"] = "regular",
        league: str = "NBA",
        top_n: int = 100,
    ) -> dict:
        """
        複数のラベルに対して連続試合記録を一括取得

        Parameters
        ----------
        labels : List[str]
            対象の列名リスト
        game_type : str
            試合タイプ
        league : str
            リーグ
        top_n : int
            各ラベルの上位N件

        Returns
        -------
        dict
            {label: DataFrame} の辞書
        """
        results = {}
        for label in labels:
            results[label] = self.get_consecutive_games(
                label, game_type, league, top_n
            )
        return results

    # =========================================================================
    # 2. 到達試合数の分析
    # =========================================================================

    def get_games_to_reach(
        self,
        label: str,
        threshold: int,
        game_type: Literal["regular", "playoff", "final", "all"] = "regular",
        league: str = "NBA",
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        累計〇〇に到達するまでの試合数ランキングを取得

        Parameters
        ----------
        label : str
            累計対象の列名（例: "PTS", "TRB", "AST"）
        threshold : int
            到達閾値（例: 10000, 5000）
        game_type : str
            試合タイプ
        league : str
            リーグ
        top_n : int
            上位N件

        Returns
        -------
        pd.DataFrame
            到達に必要な試合数が少ない順のランキング
        """
        data = self._filter_by_game_type(game_type, league)
        data = data[data["Played"] == 1]

        # 選手ごとに到達試合数を計算
        result = data.groupby("playerName").apply(
            lambda x: self._count_games_to_reach(x[label].values, threshold),
            include_groups=False
        ).reset_index()
        result.columns = ["playerName", "Games"]

        # NaN（未到達）を除外し、少ない順にソート
        result = result.dropna()
        result["Games"] = result["Games"].astype(int)
        result = result.sort_values("Games", ascending=True).head(top_n)

        return result

    @staticmethod
    def _count_games_to_reach(values: np.ndarray, threshold: int) -> Optional[int]:
        """累計がthresholdに到達するまでの試合数を計算"""
        cumsum = 0
        for idx, val in enumerate(values):
            cumsum += val
            if cumsum >= threshold:
                return idx + 1
        return None  # 未到達

    def get_multiple_thresholds_reach(
        self,
        label: str,
        thresholds: List[int],
        game_type: Literal["regular", "playoff", "final", "all"] = "regular",
        league: str = "NBA",
        top_n: int = 100,
    ) -> dict:
        """
        1つのラベルに対して複数の閾値で到達試合数を一括取得

        Parameters
        ----------
        label : str
            対象の列名
        thresholds : List[int]
            閾値のリスト
        game_type : str
            試合タイプ
        league : str
            リーグ
        top_n : int
            各閾値の上位N件

        Returns
        -------
        dict
            {threshold: DataFrame} の辞書
        """
        results = {}
        for thresh in thresholds:
            results[thresh] = self.get_games_to_reach(
                label, thresh, game_type, league, top_n
            )
        return results

    # =========================================================================
    # 3. 年間スタッツ達成回数の分析
    # =========================================================================

    def get_season_achievement_count(
        self,
        label: str,
        threshold: int,
        league: str = "NBA",
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        年間〇〇達成回数のランキングを取得

        Parameters
        ----------
        label : str
            対象の列名（例: "PTS", "Win"）
        threshold : int
            達成閾値（例: 2000 for 年間2000得点）
        league : str
            リーグ
        top_n : int
            上位N件

        Returns
        -------
        pd.DataFrame
            達成シーズン数のランキング
        """
        # レギュラーシーズンのみ
        data = self.df[
            (self.df["League"] == league) &
            (self.df["isRegular"] == 1) &
            (self.df["Played"] == 1)
        ]

        # シーズンごとに集計
        season_totals = data.groupby(
            ["playerName", "seasonStartYear"]
        )[label].sum().reset_index()

        # 閾値達成フラグ
        col_name = f"{threshold}+{label}"
        season_totals[col_name] = np.where(
            season_totals[label] >= threshold, 1, 0
        )

        # 選手ごとに達成回数を集計
        result = season_totals.groupby("playerName")[col_name].sum().reset_index()
        result = result.sort_values(col_name, ascending=False).head(top_n)

        return result

    def get_season_multi_achievement(
        self,
        conditions: dict,
        league: str = "NBA",
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        複数条件を同時に満たすシーズン数のランキング

        Parameters
        ----------
        conditions : dict
            {label: threshold} の辞書
            例: {"FG": 500, "3P": 100, "FT": 300}
        league : str
            リーグ
        top_n : int
            上位N件

        Returns
        -------
        pd.DataFrame
            全条件を達成したシーズン数のランキング
        """
        data = self.df[
            (self.df["League"] == league) &
            (self.df["isRegular"] == 1) &
            (self.df["Played"] == 1)
        ]

        labels = list(conditions.keys())
        thresholds = list(conditions.values())

        # シーズンごとに集計
        season_totals = data.groupby(
            ["playerName", "seasonStartYear"]
        )[labels].sum().reset_index()

        # 条件名を生成
        col_name = "-".join([f"{t}{l}" for l, t in conditions.items()])

        # 全条件達成フラグ
        mask = pd.Series([True] * len(season_totals))
        for label, thresh in conditions.items():
            mask &= (season_totals[label] >= thresh)
        season_totals[col_name] = mask.astype(int)

        # 選手ごとに達成回数を集計
        result = season_totals.groupby("playerName")[col_name].sum().reset_index()
        result = result.sort_values(col_name, ascending=False).head(top_n)

        return result

    # =========================================================================
    # 4. 年齢別記録の分析
    # =========================================================================

    def get_ranking_by_age(
        self,
        label: str,
        max_age: Optional[int] = None,
        min_age: Optional[int] = None,
        min_games: int = 1,
        aggfunc: str = "sum",
        league: str = "NBA",
        game_type: Literal["regular", "playoff", "final", "all"] = "regular",
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        任意の年齢範囲でのランキングを取得

        Parameters
        ----------
        label : str
            対象の列名（例: "PTS", "TRB", "AST"）
        max_age : int, optional
            最大年齢（この年齢以下、例: 25なら25歳以下）
        min_age : int, optional
            最小年齢（この年齢以上、例: 30なら30歳以上）
        min_games : int
            最低試合数
        aggfunc : str
            集計関数（"sum" or "mean"）
        league : str
            リーグ
        game_type : str
            "regular", "playoff", "all"
        top_n : int
            上位N件

        Returns
        -------
        pd.DataFrame
            ランキング（playerName, label, Games列）

        Examples
        --------
        # 25歳以下の通算得点TOP100
        analyzer.get_ranking_by_age("PTS", max_age=25, top_n=100)

        # 35歳以上の通算アシストTOP50
        analyzer.get_ranking_by_age("AST", min_age=35, top_n=50)

        # 20-25歳の平均得点TOP20
        analyzer.get_ranking_by_age("PTS", min_age=20, max_age=25, aggfunc="mean", top_n=20)
        """
        if "age_at_game" not in self.df.columns:
            raise ValueError(
                "年齢列が追加されていません。"
                "NBADataLoader.add_age_columns()を実行してください"
            )

        # ベースフィルタリング
        data = self._filter_by_game_type(game_type, league)
        data = data[data["Played"] == 1]

        # 年齢でフィルタリング（試合日時点の正確な年齢を使用）
        if max_age is not None:
            data = data[data["age_at_game"] <= max_age]
        if min_age is not None:
            data = data[data["age_at_game"] >= min_age]

        if len(data) == 0:
            return pd.DataFrame(columns=["playerName", label, "Games"])

        # 集計
        if aggfunc == "sum":
            result = data.groupby("playerName").agg({
                label: "sum",
                "game_id": "count"
            }).reset_index()
            result[label] = result[label].astype(int)
        else:
            result = data.groupby("playerName").agg({
                label: "mean",
                "game_id": "count"
            }).reset_index()
            result[label] = result[label].round(1)

        result = result.rename(columns={"game_id": "Games"})

        # 最低試合数でフィルタ
        result = result[result["Games"] >= min_games]

        # ソート
        result = result.sort_values(label, ascending=False).head(top_n)

        return result

    def get_age_based_ranking(
        self,
        label: str,
        age_threshold: int = 30,
        is_over: bool = True,
        min_games: int = 50,
        aggfunc: str = "sum",
        league: str = "NBA",
        game_type: Literal["regular", "playoff", "final", "all"] = "regular",
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        年齢ベースのランキングを取得（後方互換性のため維持）

        Parameters
        ----------
        label : str
            対象の列名
        age_threshold : int
            年齢閾値
        is_over : bool
            True: 閾値以上、False: 閾値未満
        min_games : int
            最低試合数
        aggfunc : str
            集計関数（"sum" or "mean"）
        league : str
            リーグ
        game_type : str
            "regular", "playoff", "all"
        top_n : int
            上位N件

        Returns
        -------
        pd.DataFrame
            ランキング
        """
        # 新しいメソッドに委譲
        if is_over:
            return self.get_ranking_by_age(
                label=label,
                min_age=age_threshold,
                min_games=min_games,
                aggfunc=aggfunc,
                league=league,
                game_type=game_type,
                top_n=top_n,
            )
        else:
            return self.get_ranking_by_age(
                label=label,
                max_age=age_threshold - 1,
                min_games=min_games,
                aggfunc=aggfunc,
                league=league,
                game_type=game_type,
                top_n=top_n,
            )

    def get_season_age_ranking(
        self,
        label: str,
        min_age: int = 30,
        min_games: int = 50,
        aggfunc: str = "sum",
        league: str = "NBA",
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        シーズンごとの年齢別ランキング

        Parameters
        ----------
        label : str
            対象の列名
        min_age : int
            最低年齢
        min_games : int
            最低試合数
        aggfunc : str
            集計関数
        league : str
            リーグ
        top_n : int
            上位N件

        Returns
        -------
        pd.DataFrame
            シーズン・選手・年齢ごとのランキング
        """
        if "age_at_season_end" not in self.df.columns:
            raise ValueError("年齢列が追加されていません")

        data = self.df[
            (self.df["age_at_season_end"] >= min_age) &
            (self.df["League"] == league) &
            (self.df["isRegular"] == 1) &
            (self.df["Played"] == 1)
        ]

        # 集計
        if aggfunc == "sum":
            result = data.groupby(
                ["season", "playerName", "age_at_season_end"]
            ).agg({label: "sum", "game_id": "count"}).reset_index()
        else:
            result = data.groupby(
                ["season", "playerName", "age_at_season_end"]
            ).agg({label: "mean", "game_id": "count"}).reset_index()

        result = result.rename(columns={
            "age_at_season_end": "age",
            "game_id": "Games"
        })
        result["age"] = result["age"].apply(lambda x: f"{int(x)}歳")

        # 最低試合数でフィルタ
        result = result[result["Games"] >= min_games]

        # ソート
        result = result.sort_values(label, ascending=False).head(top_n)

        return result

    # =========================================================================
    # 5. チーム内最高得点者分析
    # =========================================================================

    def get_team_scoring_leader(
        self,
        season_range: Optional[tuple] = None,
        game_type: Literal["regular", "playoff", "final", "all"] = "regular",
    ) -> pd.DataFrame:
        """
        各試合でチーム内最高得点だった選手を取得

        Parameters
        ----------
        season_range : tuple, optional
            (開始シーズン年, 終了シーズン年)
        game_type : str
            試合タイプ

        Returns
        -------
        pd.DataFrame
            各試合のチーム内最高得点者
        """
        data = self._filter_by_game_type(game_type)

        if season_range:
            data = data[
                data["seasonStartYear"].between(season_range[0], season_range[1])
            ]

        # チーム内で最高得点の行を抽出
        idx = data.groupby(["game_id", "teamName"])["PTS"].idxmax()
        result = data.loc[idx][["game_id", "teamName", "playerName", "PTS"]]

        return result

    def get_team_scoring_leader_count(
        self,
        season_range: Optional[tuple] = None,
        game_type: Literal["regular", "playoff", "final", "all"] = "regular",
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        チーム内最高得点回数のランキング

        Parameters
        ----------
        season_range : tuple, optional
            (開始シーズン年, 終了シーズン年)
        game_type : str
            試合タイプ
        top_n : int
            上位N件

        Returns
        -------
        pd.DataFrame
            最高得点回数のランキング
        """
        leaders = self.get_team_scoring_leader(season_range, game_type)

        result = leaders.groupby("playerName").size().reset_index(name="Games")
        result = result.sort_values("Games", ascending=False).head(top_n)

        return result

    # =========================================================================
    # 6. n試合スパン分析
    # =========================================================================

    def get_n_game_span_ranking(
        self,
        label: str,
        n_games: int = 2,
        game_type: Literal["regular", "playoff", "final", "all"] = "regular",
        league: str = "NBA",
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        連続n試合スパンでの合計ランキングを取得

        Parameters
        ----------
        label : str
            対象の列名（例: "PTS", "TRB", "AST"）
        n_games : int
            連続試合数（例: 2なら連続2試合の合計）
        game_type : str
            "regular", "playoff", "final", "all"
        league : str
            リーグ
        top_n : int
            上位N件

        Returns
        -------
        pd.DataFrame
            選手ごとの最大n試合スパン合計ランキング

        Examples
        --------
        # 連続2試合での最高合計得点ランキング
        analyzer.get_n_game_span_ranking("PTS", n_games=2, top_n=50)

        # 連続5試合での最高合計アシストランキング
        analyzer.get_n_game_span_ranking("AST", n_games=5, top_n=50)
        """
        data = self._filter_by_game_type(game_type, league)
        data = data[data["Played"] == 1].copy()

        # 選手ごとに日時順でソート
        data = data.sort_values(["playerName", "datetime"])

        # 選手ごとにrolling sumを計算
        col_name = f"{n_games}game_sum"
        data[col_name] = data.groupby("playerName")[label].transform(
            lambda x: x.rolling(window=n_games, min_periods=n_games).sum()
        )

        # 各選手の最大値を取得
        result = data.groupby("playerName").agg({
            col_name: "max"
        }).reset_index()

        result = result.dropna()
        result[col_name] = result[col_name].astype(int)
        result = result.sort_values(col_name, ascending=False).head(top_n)
        result = result.rename(columns={col_name: label})

        return result

    def get_n_game_span_detail(
        self,
        player_name: str,
        label: str,
        n_games: int = 2,
        game_type: Literal["regular", "playoff", "final", "all"] = "regular",
        league: str = "NBA",
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        特定選手のn試合スパン詳細を取得

        Parameters
        ----------
        player_name : str
            選手名
        label : str
            対象の列名
        n_games : int
            連続試合数
        game_type : str
            "regular", "playoff", "final", "all"
        league : str
            リーグ
        top_n : int
            上位N件

        Returns
        -------
        pd.DataFrame
            選手のn試合スパン詳細（日付、対戦相手、各試合のスタッツなど）
        """
        data = self._filter_by_game_type(game_type, league)
        data = data[(data["Played"] == 1) & (data["playerName"] == player_name)].copy()

        # 日時順でソート
        data = data.sort_values("datetime").reset_index(drop=True)

        # rolling sumを計算
        col_name = f"{n_games}game_sum"
        data[col_name] = data[label].rolling(window=n_games, min_periods=n_games).sum()

        # NaNを除外してソート
        data = data.dropna(subset=[col_name])
        data = data.sort_values(col_name, ascending=False).head(top_n)

        # 必要な列を選択
        result_cols = ["datetime", "season", "teamName", label, col_name]
        available_cols = [c for c in result_cols if c in data.columns]

        return data[available_cols]

    def get_multiple_span_rankings(
        self,
        label: str,
        spans: List[int],
        game_type: Literal["regular", "playoff", "final", "all"] = "regular",
        league: str = "NBA",
        top_n: int = 100,
    ) -> dict:
        """
        複数のスパンに対してランキングを一括取得

        Parameters
        ----------
        label : str
            対象の列名
        spans : List[int]
            スパンのリスト（例: [2, 3, 5, 10]）
        game_type : str
            試合タイプ
        league : str
            リーグ
        top_n : int
            各スパンの上位N件

        Returns
        -------
        dict
            {span: DataFrame} の辞書
        """
        results = {}
        for n in spans:
            results[n] = self.get_n_game_span_ranking(
                label=label,
                n_games=n,
                game_type=game_type,
                league=league,
                top_n=top_n,
            )
        return results

    def get_n_game_span_all_records(
        self,
        label: str,
        n_games: int = 2,
        game_type: Literal["regular", "playoff", "final", "all"] = "regular",
        league: str = "NBA",
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        連続n試合スパンでの全記録ランキング（選手重複あり）

        Parameters
        ----------
        label : str
            対象の列名（例: "PTS", "TRB", "AST"）
        n_games : int
            連続試合数
        game_type : str
            "regular", "playoff", "final", "all"
        league : str
            リーグ
        top_n : int
            上位N件

        Returns
        -------
        pd.DataFrame
            全記録ランキング（同じ選手が複数回登場可能）

        Examples
        --------
        # 連続5試合での全記録得点ランキング
        analyzer.get_n_game_span_all_records("PTS", n_games=5, top_n=100)
        """
        data = self._filter_by_game_type(game_type, league)
        data = data[data["Played"] == 1].copy()

        # 選手ごとに日時順でソート
        data = data.sort_values(["playerName", "datetime"])

        # 選手ごとにrolling sumを計算
        col_name = f"{n_games}game_sum"
        data[col_name] = data.groupby("playerName")[label].transform(
            lambda x: x.rolling(window=n_games, min_periods=n_games).sum()
        )

        # NaNを除外
        data = data.dropna(subset=[col_name])
        data[col_name] = data[col_name].astype(int)

        # 全記録をソートして上位を取得
        result = data.sort_values(col_name, ascending=False).head(top_n)

        # 必要な列を選択
        result = result[["playerName", "datetime", "season", "teamName", col_name]].copy()
        result = result.rename(columns={col_name: label})

        return result

    # =========================================================================
    # ヘルパーメソッド
    # =========================================================================

    def _filter_by_game_type(
        self,
        game_type: Literal["regular", "playoff", "final", "all"],
        league: str = "NBA",
    ) -> pd.DataFrame:
        """試合タイプでフィルタリング"""
        data = self.df[self.df["League"] == league]

        if game_type == "regular":
            data = data[data["isRegular"] == 1]
        elif game_type == "playoff":
            data = data[(data["isRegular"] == 0) & (data.get("isPlayin", 0) == 0)]
        elif game_type == "final":
            data = data[data["isFinal"] == 1]

        return data

    # =========================================================================
    # 7. デュエル（両チームトップスコアラー対決）分析
    # =========================================================================

    def get_duel_ranking(
        self,
        games_df: pd.DataFrame,
        label: str = "PTS",
        game_type: Literal["regular", "playoff", "final", "all"] = "final",
        min_total: int = 0,
        player1: Optional[str] = None,
        player2: Optional[str] = None,
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        両チームのトップスコアラー対決ランキングを取得

        各試合で両チームのトップスコアラー同士の合計スタッツが高い試合をランキング

        Parameters
        ----------
        games_df : pd.DataFrame
            試合情報データフレーム
        label : str
            対象スタッツ（例: "PTS"）
        game_type : str
            "regular", "playoff", "final", "all"
        min_total : int
            合計スタッツの最低値
        player1 : str, optional
            特定選手1が含まれる対決のみ
        player2 : str, optional
            特定選手2が含まれる対決のみ（player1と両方指定で直接対決）
        top_n : int
            上位N件

        Returns
        -------
        pd.DataFrame
            対決ランキング（Players, Score, TotalPTS, MatchUp等）

        Examples
        --------
        # ファイナル史上最高の得点対決
        analyzer.get_duel_ranking(games_df, game_type="final", top_n=50)

        # コービーが含まれる対決
        analyzer.get_duel_ranking(games_df, player1="Kobe Bryant", top_n=50)

        # コービー vs レブロンの直接対決
        analyzer.get_duel_ranking(games_df, player1="Kobe Bryant", player2="LeBron James", top_n=50)
        """
        data = self._filter_by_game_type(game_type)
        data = data[data["Played"] == 1].copy()

        if len(data) == 0:
            return pd.DataFrame()

        # 各試合・チームごとにトップスコアラーを特定
        idx = data.groupby(["game_id", "teamName"])[label].idxmax()
        top_scorers = data.loc[idx][["game_id", "teamName", "playerName", label, "Win"]].copy()

        # 試合ごとに2行（両チーム）をグループ化
        duel = top_scorers.groupby("game_id").agg({
            label: ["sum", lambda x: " - ".join(map(str, x.astype(int).tolist()))],
            "playerName": lambda x: " vs ".join(x.tolist()),
            "teamName": lambda x: " vs ".join(x.tolist()),
        }).reset_index()

        duel.columns = ["game_id", f"Total{label}", "Score", "Players", "Teams"]

        # 試合情報をマージ
        duel = duel.merge(
            games_df[["game_id", "datetime", "awayTeam", "pointsAway", "homeTeam", "pointsHome", "seasonStartYear"]],
            on="game_id"
        )

        # マッチアップ情報を作成
        duel["MatchUp"] = duel["awayTeam"] + " @ " + duel["homeTeam"]
        duel["GameScore"] = duel["pointsAway"].astype(int).astype(str) + "-" + duel["pointsHome"].astype(int).astype(str)
        duel["Season"] = duel["seasonStartYear"].astype(str) + "-" + (duel["seasonStartYear"] + 1).astype(str)

        # 最低スタッツフィルタ
        total_col = f"Total{label}"
        duel = duel[duel[total_col] >= min_total]

        # 特定選手フィルタ
        if player1:
            duel = duel[duel["Players"].str.contains(player1, na=False)]
        if player2:
            duel = duel[duel["Players"].str.contains(player2, na=False)]

        # ソート
        duel = duel.sort_values(total_col, ascending=False).reset_index(drop=True)
        duel["Rank"] = duel.index + 1

        # 出力列を整理
        result = duel[[
            "Rank", "datetime", "Season", "Players", "Score", total_col,
            "MatchUp", "GameScore"
        ]].head(top_n)

        # playerName列を追加（グラフ表示用）
        result = result.rename(columns={"Players": "playerName"})

        return result

    # =========================================================================
    # 8. 条件付き達成回数分析
    # =========================================================================

    def get_filtered_achievement_count(
        self,
        count_column: str,
        count_threshold: int,
        filter_column: Optional[str] = None,
        filter_op: Optional[str] = None,
        filter_value: Optional[Union[int, float]] = None,
        game_type: Literal["regular", "playoff", "final", "all"] = "regular",
        league: str = "NBA",
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        条件付き達成回数ランキングを取得

        特定の条件でフィルタした上で、閾値を超えた回数をカウント

        Parameters
        ----------
        count_column : str
            カウント対象列（例: "PTS", "AST"）
        count_threshold : int
            カウント閾値（例: 30 → 30以上をカウント）
        filter_column : str, optional
            フィルタ対象列（例: "FTA", "TOV"）
        filter_op : str, optional
            比較演算子（"eq", "ne", "lt", "le", "gt", "ge"）
        filter_value : int/float, optional
            フィルタ値（例: 0）
        game_type : str
            "regular", "playoff", "final", "all"
        league : str
            リーグ
        top_n : int
            上位N件

        Returns
        -------
        pd.DataFrame
            条件付き達成回数ランキング

        Examples
        --------
        # FTA=0で30得点以上の回数
        analyzer.get_filtered_achievement_count(
            count_column="PTS", count_threshold=30,
            filter_column="FTA", filter_op="eq", filter_value=0
        )

        # TOV=0で10アシスト以上の回数
        analyzer.get_filtered_achievement_count(
            count_column="AST", count_threshold=10,
            filter_column="TOV", filter_op="eq", filter_value=0
        )
        """
        data = self._filter_by_game_type(game_type, league)
        data = data[data["Played"] == 1].copy()

        # フィルタ条件を適用
        if filter_column and filter_op and filter_value is not None:
            if filter_column not in data.columns:
                raise ValueError(f"列 '{filter_column}' が存在しません")

            ops = {
                "eq": lambda x, v: x == v,
                "ne": lambda x, v: x != v,
                "lt": lambda x, v: x < v,
                "le": lambda x, v: x <= v,
                "gt": lambda x, v: x > v,
                "ge": lambda x, v: x >= v,
            }
            if filter_op not in ops:
                raise ValueError(f"不正な演算子: {filter_op}")

            data = data[ops[filter_op](data[filter_column], filter_value)]

        if len(data) == 0:
            return pd.DataFrame(columns=["playerName", "Count"])

        # カウント対象列が存在するか確認
        if count_column not in data.columns:
            raise ValueError(f"列 '{count_column}' が存在しません")

        # 閾値以上の試合をカウント
        data["_achieved"] = (data[count_column] >= count_threshold).astype(int)

        result = data.groupby("playerName")["_achieved"].sum().reset_index()
        result.columns = ["playerName", "Count"]

        # 0回を除外してソート
        result = result[result["Count"] > 0]
        result = result.sort_values("Count", ascending=False).head(top_n)

        return result

    def filter_data(
        self,
        league: str = "NBA",
        is_regular: Optional[bool] = None,
        played_only: bool = True,
        season_range: Optional[tuple] = None,
        players: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        汎用フィルタリング

        Parameters
        ----------
        league : str
            リーグ
        is_regular : bool, optional
            レギュラーシーズンかどうか
        played_only : bool
            出場試合のみ
        season_range : tuple, optional
            (開始年, 終了年)
        players : List[str], optional
            対象選手リスト

        Returns
        -------
        pd.DataFrame
            フィルタリング後のデータ
        """
        data = self.df.copy()

        data = data[data["League"] == league]

        if is_regular is not None:
            data = data[data["isRegular"] == (1 if is_regular else 0)]

        if played_only:
            data = data[data["Played"] == 1]

        if season_range:
            data = data[
                data["seasonStartYear"].between(season_range[0], season_range[1])
            ]

        if players:
            data = data[data["playerName"].isin(players)]

        return data

"""
NBA統計分析モジュール（Polars版）

各種分析関数を提供:
- 連続試合記録（〇〇得点連続試合など）
- 到達試合数（累計〇〇ポイント到達に必要な試合数）
- 年間達成回数（年間〇〇得点達成回数）
- 年齢別記録（30歳までの記録など）
"""

import polars as pl
import pandas as pd
from typing import List, Optional, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def _group_by(df: pl.DataFrame, by):
    """Polarsバージョン互換のgroup_by"""
    if hasattr(df, 'group_by'):
        return df.group_by(by)
    return df.groupby(by)


def _map_elements(expr, func, return_dtype):
    """Polarsバージョン互換のmap_elements"""
    if hasattr(expr, 'map_elements'):
        return expr.map_elements(func, return_dtype=return_dtype)
    return expr.apply(func)


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
    """NBA統計データの分析を行うクラス（Polars版）"""

    def __init__(
        self,
        df: pl.DataFrame,
        exclude_duplicate_names: bool = True,
        exclude_players: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        df : pl.DataFrame
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
            self.df = df.filter(~pl.col("playerName").is_in(self._exclude_players))
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
        data = data.filter(pl.col("Played") == 1)

        # 選手ごとに連続記録を計算
        result = (
            _group_by(data.sort(["playerName", "datetime"]), "playerName")
            .agg(
                _map_elements(
                    pl.col(label),
                    self._count_max_consecutive_list,
                    return_dtype=pl.Int64
                ).alias(label)
            )
            .sort(label, descending=True)
            .head(top_n)
        )

        return result.to_pandas()

    @staticmethod
    def _count_max_consecutive_list(values: list) -> int:
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
        """
        data = self._filter_by_game_type(game_type, league)
        data = data.filter(pl.col("Played") == 1)

        # 選手ごとに到達試合数を計算
        def count_games_to_reach(values: list) -> Optional[int]:
            cumsum = 0
            for idx, val in enumerate(values):
                if val is not None:
                    cumsum += val
                if cumsum >= threshold:
                    return idx + 1
            return None

        result = (
            _group_by(data.sort(["playerName", "datetime"]), "playerName")
            .agg(
                _map_elements(
                    pl.col(label),
                    count_games_to_reach,
                    return_dtype=pl.Int64
                ).alias("Games")
            )
            .filter(pl.col("Games").is_not_null())
            .sort("Games")
            .head(top_n)
        )

        return result.to_pandas()

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
        """
        # レギュラーシーズンのみ
        data = self.df.filter(
            (pl.col("League") == league) &
            (pl.col("isRegular") == 1) &
            (pl.col("Played") == 1)
        )

        col_name = f"{threshold}+{label}"

        # シーズンごとに集計
        season_totals = (
            _group_by(data, ["playerName", "seasonStartYear"])
            .agg(pl.col(label).sum())
        )

        # 閾値達成フラグ
        season_totals = season_totals.with_columns(
            pl.when(pl.col(label) >= threshold).then(1).otherwise(0).alias(col_name)
        )

        # 選手ごとに達成回数を集計
        result = (
            _group_by(season_totals, "playerName")
            .agg(pl.col(col_name).sum())
            .sort(col_name, descending=True)
            .head(top_n)
        )

        return result.to_pandas()

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
        """
        if "age_at_game" not in self.df.columns:
            raise ValueError(
                "年齢列が追加されていません。"
                "NBADataLoader.add_age_columns()を実行してください"
            )

        # ベースフィルタリング
        data = self._filter_by_game_type(game_type, league)
        data = data.filter(pl.col("Played") == 1)

        # 年齢でフィルタリング
        if max_age is not None:
            data = data.filter(pl.col("age_at_game") <= max_age)
        if min_age is not None:
            data = data.filter(pl.col("age_at_game") >= min_age)

        if len(data) == 0:
            return pd.DataFrame(columns=["playerName", label, "Games"])

        # 集計
        if aggfunc == "sum":
            result = (
                _group_by(data, "playerName")
                .agg([
                    pl.col(label).sum().cast(pl.Int64),
                    pl.col("game_id").count().alias("Games"),
                ])
            )
        else:
            result = (
                _group_by(data, "playerName")
                .agg([
                    pl.col(label).mean().round(1),
                    pl.col("game_id").count().alias("Games"),
                ])
            )

        # 最低試合数でフィルタ
        result = result.filter(pl.col("Games") >= min_games)

        # ソート
        result = result.sort(label, descending=True).head(top_n)

        return result.to_pandas()

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
        """
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
        """
        data = self._filter_by_game_type(game_type, league)
        data = data.filter(pl.col("Played") == 1)

        # 選手ごとに日時順でソート
        data = data.sort(["playerName", "datetime"])

        # 選手ごとにrolling sumを計算
        data = data.with_columns(
            pl.col(label).rolling_sum(window_size=n_games, min_periods=n_games)
            .over("playerName")
            .alias("_rolling_sum")
        )

        # 各選手の最大値を取得
        result = (
            _group_by(data, "playerName")
            .agg(pl.col("_rolling_sum").max().alias(label))
            .filter(pl.col(label).is_not_null())
            .with_columns(pl.col(label).cast(pl.Int64))
            .sort(label, descending=True)
            .head(top_n)
        )

        return result.to_pandas()

    # =========================================================================
    # 7. デュエル（両チームトップスコアラー対決）分析
    # =========================================================================

    def get_duel_ranking(
        self,
        games_df: pl.DataFrame,
        label: str = "PTS",
        game_type: Literal["regular", "playoff", "final", "all"] = "final",
        min_total: int = 0,
        player1: Optional[str] = None,
        player2: Optional[str] = None,
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        両チームのトップスコアラー対決ランキングを取得
        """
        data = self._filter_by_game_type(game_type)
        data = data.filter(pl.col("Played") == 1)

        if len(data) == 0:
            return pd.DataFrame()

        # 各試合・チームごとにトップスコアラーを特定
        top_scorers = (
            _group_by(data.sort(label, descending=True), ["game_id", "teamName"])
            .first()
            .select(["game_id", "teamName", "playerName", label, "Win"])
        )

        # 試合ごとに集計
        duel = (
            _group_by(top_scorers, "game_id")
            .agg([
                pl.col(label).sum().alias(f"Total{label}"),
                pl.col(label).cast(pl.Utf8).str.concat(" - ").alias("Score"),
                pl.col("playerName").str.concat(" vs ").alias("Players"),
                pl.col("teamName").str.concat(" vs ").alias("Teams"),
            ])
        )

        # games_dfがPolarsでない場合は変換
        if isinstance(games_df, pd.DataFrame):
            games_df = pl.from_pandas(games_df)

        # 試合情報をマージ
        game_cols = ["game_id", "datetime", "awayTeam", "pointsAway", "homeTeam", "pointsHome", "seasonStartYear"]
        available_game_cols = [c for c in game_cols if c in games_df.columns]
        duel = duel.join(games_df.select(available_game_cols), on="game_id")

        # マッチアップ情報を作成
        duel = duel.with_columns([
            (pl.col("awayTeam") + " @ " + pl.col("homeTeam")).alias("MatchUp"),
            (pl.col("pointsAway").cast(pl.Int64).cast(pl.Utf8) + "-" +
             pl.col("pointsHome").cast(pl.Int64).cast(pl.Utf8)).alias("GameScore"),
            (pl.col("seasonStartYear").cast(pl.Int64).cast(pl.Utf8) + "-" +
             (pl.col("seasonStartYear").cast(pl.Int64) + 1).cast(pl.Utf8)).alias("Season"),
        ])

        # 最低スタッツフィルタ
        total_col = f"Total{label}"
        duel = duel.filter(pl.col(total_col) >= min_total)

        # 特定選手フィルタ
        if player1:
            duel = duel.filter(pl.col("Players").str.contains(player1))
        if player2:
            duel = duel.filter(pl.col("Players").str.contains(player2))

        # ソート
        duel = duel.sort(total_col, descending=True)

        # ランク追加
        duel = duel.with_row_index("Rank", offset=1)

        # 出力列を整理
        result = duel.select([
            "Rank", "datetime", "Season", "Players", "Score", total_col,
            "MatchUp", "GameScore"
        ]).head(top_n)

        # playerName列を追加（グラフ表示用）
        result = result.rename({"Players": "playerName"})

        return result.to_pandas()

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
        """
        data = self._filter_by_game_type(game_type, league)
        data = data.filter(pl.col("Played") == 1)

        # フィルタ条件を適用
        if filter_column and filter_op and filter_value is not None:
            if filter_column not in data.columns:
                raise ValueError(f"列 '{filter_column}' が存在しません")

            if filter_op == "eq":
                data = data.filter(pl.col(filter_column) == filter_value)
            elif filter_op == "ne":
                data = data.filter(pl.col(filter_column) != filter_value)
            elif filter_op == "lt":
                data = data.filter(pl.col(filter_column) < filter_value)
            elif filter_op == "le":
                data = data.filter(pl.col(filter_column) <= filter_value)
            elif filter_op == "gt":
                data = data.filter(pl.col(filter_column) > filter_value)
            elif filter_op == "ge":
                data = data.filter(pl.col(filter_column) >= filter_value)
            else:
                raise ValueError(f"不正な演算子: {filter_op}")

        if len(data) == 0:
            return pd.DataFrame(columns=["playerName", "Count"])

        # カウント対象列が存在するか確認
        if count_column not in data.columns:
            raise ValueError(f"列 '{count_column}' が存在しません")

        # 閾値以上の試合をカウント
        data = data.with_columns(
            pl.when(pl.col(count_column) >= count_threshold).then(1).otherwise(0).alias("_achieved")
        )

        result = (
            _group_by(data, "playerName")
            .agg(pl.col("_achieved").sum().alias("Count"))
            .filter(pl.col("Count") > 0)
            .sort("Count", descending=True)
            .head(top_n)
        )

        return result.to_pandas()

    # =========================================================================
    # ヘルパーメソッド
    # =========================================================================

    def _filter_by_game_type(
        self,
        game_type: Literal["regular", "playoff", "final", "all"],
        league: str = "NBA",
    ) -> pl.DataFrame:
        """試合タイプでフィルタリング"""
        data = self.df.filter(pl.col("League") == league)

        if game_type == "regular":
            data = data.filter(pl.col("isRegular") == 1)
        elif game_type == "playoff":
            # isPlayinが存在しない場合も考慮
            if "isPlayin" in data.columns:
                data = data.filter(
                    (pl.col("isRegular") == 0) & (pl.col("isPlayin") == 0)
                )
            else:
                data = data.filter(pl.col("isRegular") == 0)
        elif game_type == "final":
            data = data.filter(pl.col("isFinal") == 1)

        return data

    def filter_data(
        self,
        league: str = "NBA",
        is_regular: Optional[bool] = None,
        played_only: bool = True,
        season_range: Optional[tuple] = None,
        players: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        汎用フィルタリング
        """
        data = self.df.filter(pl.col("League") == league)

        if is_regular is not None:
            data = data.filter(pl.col("isRegular") == (1 if is_regular else 0))

        if played_only:
            data = data.filter(pl.col("Played") == 1)

        if season_range:
            data = data.filter(
                pl.col("seasonStartYear").is_between(season_range[0], season_range[1])
            )

        if players:
            data = data.filter(pl.col("playerName").is_in(players))

        return data

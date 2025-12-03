"""分析実行モジュール - NBAAnalyzerを呼び出して結果を返す（Polars版）"""

import os
import sys
from pathlib import Path
from typing import Optional

import polars as pl
import pandas as pd
import streamlit as st

# srcモジュールへのパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import NBADataLoader
from src.analysis import NBAAnalyzer
from src.utils import merge_player_image


def get_data_dir() -> str:
    """
    データディレクトリを環境に応じて自動選択

    優先順位:
    1. 環境変数 DATA_DIR
    2. ローカルの data ディレクトリ
    """
    # 環境変数で明示的に指定されている場合
    if os.environ.get("DATA_DIR"):
        return os.environ["DATA_DIR"]

    # ローカル/Streamlit Cloud環境
    return "data"


# 利用可能な関数のマッピング
AVAILABLE_FUNCTIONS = {
    "get_ranking_by_age",
    "get_consecutive_games",
    "get_games_to_reach",
    "get_n_game_span_ranking",
    "get_season_achievement_count",
    "get_duel_ranking",
    "get_filtered_achievement_count",
}


@st.cache_resource(show_spinner=False)
def load_data():
    """
    データを読み込んでキャッシュ（Polars版）

    Returns:
        tuple: (df, analyzer, games_df) - 分析用データフレーム、NBAAnalyzerインスタンス、試合情報
    """
    data_dir = get_data_dir()
    loader = NBADataLoader(data_dir=data_dir)
    loader.load_boxscore("boxscore1946-2025.csv.gz")
    loader.load_games("games1946-2025.csv.gz")
    loader.load_player_info("Players_data_Latest.csv")

    df = loader.create_analysis_df()
    df = loader.add_age_columns(df)

    analyzer = NBAAnalyzer(df, exclude_duplicate_names=True)
    games_df = loader._games

    return df, analyzer, games_df


def execute_analysis(parsed: dict) -> tuple[Optional[pd.DataFrame], str]:
    """
    LLMが解釈した結果を元に分析を実行

    Args:
        parsed: LLMの解釈結果
            {
                "function": "get_ranking_by_age",
                "params": {"label": "PTS", "max_age": 25, ...},
                "description": "説明文"
            }

    Returns:
        tuple: (結果DataFrame, メッセージ)
        ※結果はpandas DataFrameとして返す（Streamlit/Plotly互換性のため）
    """
    func_name = parsed.get("function")
    params = parsed.get("params", {})
    description = parsed.get("description", "分析を実行しました")

    # 関数が指定されていない場合
    if func_name is None:
        return None, description

    # 許可されていない関数の場合
    if func_name not in AVAILABLE_FUNCTIONS:
        return None, f"「{func_name}」は対応していない分析タイプです"

    try:
        # データとアナライザーを取得（キャッシュ済み）
        df, analyzer, games_df = load_data()

        # パラメータのクリーンアップ
        params = _clean_params(func_name, params)

        # デュエル分析の場合はgames_dfを渡す
        if func_name == "get_duel_ranking":
            params["games_df"] = games_df

        # 関数を取得して実行
        method = getattr(analyzer, func_name)
        result = method(**params)

        # 結果が空の場合
        if result is None or len(result) == 0:
            return None, "条件に一致するデータが見つかりませんでした"

        # Polars DataFrameの場合はpandasに変換
        if isinstance(result, pl.DataFrame):
            result = result.to_pandas()

        # 選手画像をマージ
        if "playerName" in result.columns:
            image_csv = f"{get_data_dir()}/player_imageURL.csv"
            result = merge_player_image(result, image_csv=image_csv)

        return result, description

    except Exception as e:
        return None, f"分析中にエラーが発生しました: {str(e)}"


def _clean_params(func_name: str, params: dict) -> dict:
    """
    パラメータをクリーンアップ（型変換、不要パラメータの除去）
    """
    cleaned = {}

    # 各関数で許可されているパラメータ
    allowed_params = {
        "get_ranking_by_age": {
            "label", "max_age", "min_age", "min_games", "aggfunc",
            "league", "game_type", "top_n"
        },
        "get_consecutive_games": {
            "label", "game_type", "league", "top_n"
        },
        "get_games_to_reach": {
            "label", "threshold", "game_type", "league", "top_n"
        },
        "get_n_game_span_ranking": {
            "label", "n_games", "game_type", "league", "top_n"
        },
        "get_season_achievement_count": {
            "label", "threshold", "league", "top_n"
        },
        "get_duel_ranking": {
            "label", "game_type", "min_total", "player1", "player2", "top_n"
        },
        "get_filtered_achievement_count": {
            "count_column", "count_threshold", "filter_column", "filter_op",
            "filter_value", "game_type", "league", "top_n"
        },
    }

    allowed = allowed_params.get(func_name, set())

    for key, value in params.items():
        # 許可されていないパラメータはスキップ
        if key not in allowed:
            continue

        # None値はスキップ
        if value is None:
            continue

        # 型変換
        if key in ("max_age", "min_age", "min_games", "threshold", "n_games", "top_n", "min_total", "count_threshold", "filter_value"):
            try:
                value = int(value)
            except (ValueError, TypeError):
                continue

        cleaned[key] = value

    # デフォルト値の設定
    if "top_n" not in cleaned:
        cleaned["top_n"] = 10  # デフォルトはTOP10

    if "game_type" not in cleaned:
        cleaned["game_type"] = "regular"

    # aggfuncのデフォルトは"sum"（回数をカウントする場合が多いため）
    if func_name == "get_ranking_by_age" and "aggfunc" not in cleaned:
        cleaned["aggfunc"] = "sum"

    return cleaned


def get_value_column(result: pd.DataFrame, parsed: dict) -> Optional[str]:
    """
    グラフ表示用の値列を特定

    Args:
        result: 結果DataFrame
        parsed: LLMの解釈結果

    Returns:
        str: 値列の名前
    """
    # パラメータからラベルを取得
    label = parsed.get("params", {}).get("label")

    if label and label in result.columns:
        return label

    # 一般的な値列名を探す
    for col in ["TotalPTS", "PTS", "TRB", "AST", "STL", "BLK", "3P", "Win", "DD", "TD", "Games", "Count"]:
        if col in result.columns:
            return col

    # playerName以外の数値列を探す
    numeric_cols = result.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        if col != "playerName":
            return col

    return None

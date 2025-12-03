"""
ユーティリティ関数モジュール（Polars版）

- 画像URLのマージ
- ランキングデータの整形
- CSV出力
"""

import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional, List, Union


# =============================================================================
# 画像データのマージ
# =============================================================================

def merge_player_image(
    df: Union[pl.DataFrame, pd.DataFrame],
    player_col: str = "playerName",
    image_csv: str = "data/player_imageURL.csv",
) -> pd.DataFrame:
    """
    選手の画像URLをマージ

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        対象のデータフレーム
    player_col : str
        選手名の列名
    image_csv : str
        画像URLのCSVパス

    Returns
    -------
    pd.DataFrame
        画像URL列を追加したデータフレーム（pandas）
    """
    # pandasに変換
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    player_im = pd.read_csv(image_csv)

    # player_colが"playerName"でない場合はリネーム
    if player_col != "playerName" and "playerName" in player_im.columns:
        player_im = player_im.rename(columns={"playerName": player_col})

    result = df.merge(
        player_im[[player_col, "image_url"]],
        on=player_col,
        how="left"
    )

    # 画像列を先頭に移動
    if "image_url" in result.columns:
        result.insert(0, "player_image", result.pop("image_url"))

    return result


def merge_team_image(
    df: Union[pl.DataFrame, pd.DataFrame],
    team_col: str = "teamName",
    image_csv: str = "data/team_images.csv",
) -> pd.DataFrame:
    """
    チームの画像URLと略称をマージ

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        対象のデータフレーム
    team_col : str
        チーム名の列名
    image_csv : str
        画像URLのCSVパス

    Returns
    -------
    pd.DataFrame
        画像URL・略称列を追加したデータフレーム（pandas）
    """
    # pandasに変換
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    team_im = pd.read_csv(image_csv)

    # カラム名を合わせる
    if "team" in team_im.columns and team_col != "team":
        team_im = team_im.rename(columns={"team": team_col})

    result = df.merge(team_im, on=team_col, how="left")

    # 列を整形
    if "team_im" in result.columns:
        result.insert(0, "team_image", result.pop("team_im"))
    if "abbreviation" in result.columns:
        result.insert(1, "teamAbb", result.pop("abbreviation"))

    return result


# =============================================================================
# ランキングデータの整形
# =============================================================================

def format_ranking(
    df: Union[pl.DataFrame, pd.DataFrame],
    value_col: str,
    ascending: bool = False,
    add_rank: bool = True,
    rank_col: str = "Rank",
) -> pd.DataFrame:
    """
    ランキングデータを整形

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        ランキングデータ
    value_col : str
        ランキングの基準となる列
    ascending : bool
        昇順ソートかどうか
    add_rank : bool
        順位列を追加するか
    rank_col : str
        順位列の名前

    Returns
    -------
    pd.DataFrame
        整形後のデータフレーム（pandas）
    """
    # pandasに変換
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    result = df.sort_values(value_col, ascending=ascending).reset_index(drop=True)

    if add_rank:
        result.insert(0, rank_col, range(1, len(result) + 1))

    return result


def shorten_player_name(name: str, max_length: int = 15) -> str:
    """
    長い選手名を短縮

    Parameters
    ----------
    name : str
        選手名
    max_length : int
        最大文字数

    Returns
    -------
    str
        短縮された名前
    """
    if pd.isna(name):
        return name

    parts = str(name).split()
    if len(parts) < 2:
        return name

    full_name = f"{parts[0]} {parts[-1]}"
    if len(full_name) <= max_length:
        return full_name

    # ファーストネームをイニシャルに
    return f"{parts[0][0]}.{parts[-1]}"


def add_short_name_column(
    df: Union[pl.DataFrame, pd.DataFrame],
    name_col: str = "playerName",
    new_col: str = "shortName",
    max_length: int = 15,
) -> pd.DataFrame:
    """
    短縮名の列を追加

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        対象のデータフレーム
    name_col : str
        選手名の列
    new_col : str
        新しい列名
    max_length : int
        最大文字数

    Returns
    -------
    pd.DataFrame
        短縮名列を追加したデータフレーム（pandas）
    """
    # pandasに変換
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    result = df.copy()
    result[new_col] = result[name_col].apply(
        lambda x: shorten_player_name(x, max_length)
    )
    return result


# =============================================================================
# 選手名の標準化
# =============================================================================

# 同姓同名や表記揺れの選手を除外するためのリスト（必要に応じて追加）
EXCLUDE_PLAYERS = [
    "Eddie Johnson",      # 複数の選手が存在
    "George Johnson",     # 複数の選手が存在
    "Mike Dunleavy",      # 親子で同名
    "David Lee",          # 複数の選手が存在
    "Jim Paxson",
    "Larry Johnson",
    "Matt Guokas",
]


def filter_duplicate_names(
    df: Union[pl.DataFrame, pd.DataFrame],
    player_col: str = "playerName",
    exclude_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    同姓同名の選手を除外

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        対象のデータフレーム
    player_col : str
        選手名の列
    exclude_list : List[str], optional
        除外する選手名リスト（指定しない場合はデフォルトリストを使用）

    Returns
    -------
    pd.DataFrame
        除外後のデータフレーム（pandas）
    """
    # pandasに変換
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    exclude = exclude_list or EXCLUDE_PLAYERS
    return df[~df[player_col].isin(exclude)]


# 表記揺れを統一するためのマッピング
PLAYER_NAME_MAPPING = {
    "Kareem Abdul-Jabbar": "K.Abdul-Jabbar",
    "Karl-Anthony Towns": "K-Anthony Towns",
    "Micheal Ray Richardson": "M.Ray Richardson",
    "Giannis Antetokounmpo": "G.Antetokounmpo",
}


def standardize_player_names(
    df: Union[pl.DataFrame, pd.DataFrame],
    player_col: str = "playerName",
    mapping: Optional[dict] = None,
) -> pd.DataFrame:
    """
    選手名を標準化

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        対象のデータフレーム
    player_col : str
        選手名の列
    mapping : dict, optional
        変換マッピング

    Returns
    -------
    pd.DataFrame
        標準化後のデータフレーム（pandas）
    """
    # pandasに変換
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    result = df.copy()
    name_map = mapping or PLAYER_NAME_MAPPING
    result[player_col] = result[player_col].replace(name_map)
    return result


# =============================================================================
# CSV出力
# =============================================================================

def save_ranking_to_csv(
    df: Union[pl.DataFrame, pd.DataFrame],
    filename: str,
    output_dir: str = "output",
    add_images: bool = True,
    image_csv: str = "data/player_imageURL.csv",
) -> str:
    """
    ランキングデータをCSVに保存

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        保存するデータフレーム
    filename : str
        ファイル名
    output_dir : str
        出力ディレクトリ
    add_images : bool
        画像URLを追加するか
    image_csv : str
        画像URLのCSVパス

    Returns
    -------
    str
        保存したファイルパス
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # pandasに変換
    if isinstance(df, pl.DataFrame):
        result = df.to_pandas()
    else:
        result = df.copy()

    if add_images and "playerName" in result.columns:
        result = merge_player_image(result, image_csv=image_csv)

    filepath = output_path / filename
    result.to_csv(filepath, index=False)

    return str(filepath)


def batch_save_rankings(
    rankings: dict,
    output_dir: str = "output",
    prefix: str = "",
    suffix: str = "",
) -> List[str]:
    """
    複数のランキングを一括保存

    Parameters
    ----------
    rankings : dict
        {name: DataFrame} の辞書
    output_dir : str
        出力ディレクトリ
    prefix : str
        ファイル名の接頭辞
    suffix : str
        ファイル名の接尾辞

    Returns
    -------
    List[str]
        保存したファイルパスのリスト
    """
    saved_files = []
    for name, df in rankings.items():
        filename = f"{prefix}{name}{suffix}.csv"
        filepath = save_ranking_to_csv(df, filename, output_dir)
        saved_files.append(filepath)

    return saved_files

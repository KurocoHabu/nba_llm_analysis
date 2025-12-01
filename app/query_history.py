"""質問履歴の保存・取得モジュール"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# 履歴ファイルのパス
HISTORY_FILE = Path(__file__).parent.parent / "data" / "query_history.json"

# 最大保存件数
MAX_HISTORY = 100


def load_history() -> list[dict]:
    """履歴を読み込む"""
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_query(query: str, description: str, function: Optional[str] = None) -> None:
    """
    成功した質問を履歴に保存

    Args:
        query: ユーザーの質問文
        description: 分析内容の説明
        function: 実行された関数名
    """
    history = load_history()

    # 重複チェック（同じ質問は保存しない）
    existing_queries = {item["query"] for item in history}
    if query in existing_queries:
        return

    # 新しい質問を追加
    history.append({
        "query": query,
        "description": description,
        "function": function,
        "timestamp": datetime.now().isoformat(),
    })

    # 最大件数を超えたら古いものを削除
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    # 保存
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def get_recent_queries(limit: int = 10) -> list[str]:
    """
    最近の質問を取得（新しい順）

    Args:
        limit: 取得件数

    Returns:
        質問文のリスト
    """
    history = load_history()
    # 新しい順にソート
    history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return [item["query"] for item in history[:limit]]


def get_popular_queries(limit: int = 10) -> list[str]:
    """
    よく使われる質問パターンを取得
    （将来的にカウント機能を追加可能）

    Args:
        limit: 取得件数

    Returns:
        質問文のリスト
    """
    # 現時点では最近の質問を返す
    return get_recent_queries(limit)

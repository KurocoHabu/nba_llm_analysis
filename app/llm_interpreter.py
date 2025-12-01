"""Claude Haiku APIとの連携モジュール"""

import json
import streamlit as st
from anthropic import Anthropic

from .prompts import SYSTEM_PROMPT, build_messages


# Claude Haiku 4.5モデル
MODEL = "claude-haiku-4-5-20251001"


def get_client() -> Anthropic:
    """Anthropicクライアントを取得"""
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEYが設定されていません。.streamlit/secrets.tomlを確認してください。")
    return Anthropic(api_key=api_key)


def interpret_query(user_query: str) -> dict:
    """
    ユーザーの自然言語クエリを解析し、構造化されたパラメータを返す

    Args:
        user_query: ユーザーからの質問（例: "25歳以下の通算得点TOP30"）

    Returns:
        dict: {
            "function": "get_ranking_by_age",
            "params": {"label": "PTS", "max_age": 25, ...},
            "description": "25歳以下の通算得点TOP30を取得します"
        }
    """
    client = get_client()

    messages = build_messages(user_query)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        # レスポンスからテキストを抽出
        content = response.content[0].text.strip()

        # JSONをパース
        result = json.loads(content)

        # バリデーション
        if "function" not in result:
            result["function"] = None
        if "params" not in result:
            result["params"] = {}
        if "description" not in result:
            result["description"] = "分析を実行します"

        return result

    except json.JSONDecodeError as e:
        return {
            "function": None,
            "params": {},
            "description": f"応答の解析に失敗しました: {str(e)}",
            "error": True,
        }
    except Exception as e:
        return {
            "function": None,
            "params": {},
            "description": f"エラーが発生しました: {str(e)}",
            "error": True,
        }


def is_valid_interpretation(result: dict) -> bool:
    """解釈結果が有効かどうかをチェック"""
    return result.get("function") is not None and not result.get("error", False)


def generate_analysis_comment(query: str, result_df, parsed: dict) -> str:
    """
    分析結果に対する考察コメントを生成（200-300文字）

    Args:
        query: ユーザーのクエリ
        result_df: 分析結果DataFrame
        parsed: LLMの解釈結果

    Returns:
        str: 考察コメント
    """
    client = get_client()

    # 結果のTOP5を抽出
    top5 = result_df.head(5)
    if "playerName" in top5.columns:
        # 値の列を特定
        value_cols = [c for c in top5.columns if c not in ["playerName", "player_image", "Games"]]
        if value_cols:
            value_col = value_cols[0]
            top5_text = "\n".join([
                f"{i+1}. {row['playerName']}: {row[value_col]}"
                for i, (_, row) in enumerate(top5.iterrows())
            ])
        else:
            top5_text = top5.to_string(index=False)
    else:
        top5_text = top5.to_string(index=False)

    prompt = f"""以下のNBAスタッツ分析結果について、100〜200文字程度で簡潔に考察を書いてください。
NBAファン向けに興味深い観点や意外な発見を1-2点指摘してください。
マークダウン形式は使わず、プレーンテキストで書いてください。

質問: {query}
分析内容: {parsed.get('description', '')}

結果TOP5:
{top5_text}

考察（100-200文字、プレーンテキスト）:"""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception:
        return ""

"""NBA Stats Chat - Streamlitアプリ メインエントリポイント"""

import sys
import urllib.parse
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from app.styles import CUSTOM_CSS, get_plotly_theme, get_bar_color
from app.llm_interpreter import interpret_query, is_valid_interpretation, generate_analysis_comment
from app.executor import execute_analysis, get_value_column
from app.query_history import save_query, get_recent_queries


# ページ設定
st.set_page_config(
    page_title="NBA Player Analytics",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# カスタムCSS適用
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session_state():
    """セッション状態の初期化"""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sidebar():
    """サイドバーを描画"""
    with st.sidebar:
        st.markdown("## 📚 使い方")
        st.markdown("""
        質問を入力すると、NBAスタッツを分析して結果を表示します。
        """)

        st.markdown("### 💡 質問例")
        examples = [
            "25歳時点での通算得点ランキング",
            "連続ダブルダブル記録TOP20",
            "連勝記録ランキング",
            "1万得点到達までの試合数TOP15",
            "プレイオフでの40得点ゲーム回数ランキング",
            "10試合スパンでの最高合計得点",
            "35歳以上の通算アシストTOP5",
            "ゲーム別のベストデュエルランキングを見たい",
        ]
        for example in examples:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                st.session_state.pending_query = example
                st.rerun()

        st.markdown("---")

        st.markdown("### 📊 対応スタッツ")
        st.code("PTS, TRB, AST, STL, BLK, 3P, Win, DD, TD")

        st.markdown("### 🎮 試合タイプ")
        st.markdown("- `regular`: レギュラーシーズン")
        st.markdown("- `playoff`: プレイオフ")
        st.markdown("- `final`: ファイナル")
        st.markdown("- `all`: 全試合")

        # 過去の質問履歴
        recent_queries = get_recent_queries(limit=10)
        if recent_queries:
            st.markdown("---")
            st.markdown("### 📜 過去の質問")
            for q in recent_queries:
                if st.button(q, key=f"history_{q}", use_container_width=True):
                    st.session_state.pending_query = q
                    st.rerun()

        # フッター
        st.markdown("---")
        st.markdown(
            "📬 ご希望の分析メニューや機能がありましたら "
            "[nba.bcr2022@gmail.com](mailto:nba.bcr2022@gmail.com) "
            "までお気軽にお問い合わせください！"
        )


def create_bar_chart(df, value_col: str, title: str = "", max_display: int = 50, highlight_query: str = "") -> go.Figure:
    """横棒グラフを作成（スクロール対応、選手ハイライト機能付き）"""
    # 表示件数を制限
    plot_df = df.head(max_display).copy()
    n_bars = len(plot_df)

    # クエリに含まれる選手をハイライト
    highlight_color = "#FFD700"  # ゴールド
    normal_color = get_bar_color()

    # 選手名がクエリに含まれているかチェック
    def should_highlight(player_name: str) -> bool:
        if not highlight_query:
            return False
        query_lower = highlight_query.lower()
        # フルネームまたは姓・名の一部がクエリに含まれているか
        name_parts = player_name.lower().split()
        return (
            player_name.lower() in query_lower or
            any(part in query_lower for part in name_parts if len(part) > 2)
        )

    plot_df["_highlight"] = plot_df["playerName"].apply(should_highlight)

    # ランキング番号を追加（1位から順に）
    plot_df = plot_df.reset_index(drop=True)
    plot_df["_display_name"] = plot_df.apply(
        lambda row: f"{row.name + 1}. {row['playerName']}", axis=1
    )

    # 逆順にする（1位が上に来るように）
    plot_df = plot_df.iloc[::-1]

    # 色リストを作成
    colors = [highlight_color if h else normal_color for h in plot_df["_highlight"]]

    fig = px.bar(
        plot_df,
        x=value_col,
        y="_display_name",
        orientation="h",
        title=title,
        text=value_col,  # バーにラベル表示
    )

    # 色を適用
    fig.update_traces(marker_color=colors)

    # バーの高さを固定（1バーあたり30px）
    chart_height = max(600, n_bars * 30)

    # テーマ適用
    theme = get_plotly_theme()
    fig.update_layout(
        **theme,
        height=chart_height,
        showlegend=False,
        xaxis_title=value_col,
        yaxis_title="",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # ラベルのスタイル設定
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=14),
    )

    # Y軸（選手名）のフォントサイズ（白色）
    fig.update_yaxes(tickfont=dict(size=14, color="#FFFFFF"))

    return fig


def render_result(result_df, parsed: dict, msg_idx: int, comment: str = "", query: str = ""):
    """分析結果を表示"""
    value_col = get_value_column(result_df, parsed)
    func_name = parsed.get("function", "")

    # デュエル分析はテーブルのみ表示
    if func_name == "get_duel_ranking":
        st.markdown(f"**{parsed.get('description', '')}**")
        display_df = result_df.copy()
        if "player_image" in display_df.columns:
            display_df = display_df.drop(columns=["player_image"])
        st.dataframe(display_df, use_container_width=True, height=500)
    else:
        # タブで表示切り替え
        tab_chart, tab_table = st.tabs(["📊 グラフ", "📋 テーブル"])

        with tab_chart:
            if value_col and "playerName" in result_df.columns:
                fig = create_bar_chart(
                    result_df,
                    value_col,
                    title="",  # タイトルは上のコメントと重複するので削除
                    max_display=50,  # 最大50件表示
                    highlight_query=query  # クエリに含まれる選手をハイライト
                )
                # スクロール可能なコンテナでラップ
                with st.container(height=600):
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{msg_idx}")
            else:
                st.info("グラフ表示には対応していないデータ形式です")
                st.dataframe(result_df.head(20), use_container_width=True)

        with tab_table:
            # 表示用に列を整理
            display_df = result_df.copy()
            if "player_image" in display_df.columns:
                display_df = display_df.drop(columns=["player_image"])
            st.dataframe(display_df, use_container_width=True, height=400)

    # 考察コメント表示
    if comment:
        st.markdown("### 💡 考察")
        st.info(comment)

    # ボタンを横並びに（左寄せ）
    col1, col2, col3 = st.columns([1, 1, 4])

    # CSVダウンロード（クエリをファイル名に使用）
    with col1:
        csv = result_df.to_csv(index=False).encode("utf-8")
        # ファイル名に使えない文字を置換
        safe_query = query.replace("/", "_").replace("\\", "_").replace(":", "_")[:50] if query else "result"
        st.download_button(
            label="📥 CSVダウンロード",
            data=csv,
            file_name=f"{safe_query}.csv",
            mime="text/csv",
            key=f"download_{msg_idx}",
        )

    # Xシェアボタン
    with col2:
        # シェア用テキストを作成（TOP3を含む）
        share_text = f"{query}\n\n"
        if "playerName" in result_df.columns and value_col:
            for i, row in result_df.head(3).iterrows():
                rank = result_df.index.get_loc(i) + 1
                share_text += f"{rank}. {row['playerName']}: {row[value_col]}\n"
        share_text += "\n#NBA #NBAStats"

        # URLエンコード
        encoded_text = urllib.parse.quote(share_text)
        twitter_url = f"https://twitter.com/intent/tweet?text={encoded_text}"

        st.markdown(
            f'<a href="{twitter_url}" target="_blank" style="'
            'display: inline-block; padding: 0.5rem 1rem; '
            'background-color: #1DA1F2; color: white; '
            'text-decoration: none; border-radius: 0.5rem; '
            'font-weight: 600;">𝕏 シェア</a>',
            unsafe_allow_html=True
        )


def process_query(query: str):
    """クエリを処理"""
    # ユーザーメッセージを追加
    st.session_state.messages.append({
        "role": "user",
        "content": query,
    })

    # LLMで解釈
    with st.spinner("🏀 分析中..."):
        parsed = interpret_query(query)

        if is_valid_interpretation(parsed):
            # 分析実行
            result, message = execute_analysis(parsed)

            if result is not None:
                # 考察コメントを生成
                comment = generate_analysis_comment(query, result, parsed)

                # 成功した質問を履歴に保存
                save_query(
                    query=query,
                    description=message,
                    function=parsed.get("function")
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": message,
                    "result": result,
                    "parsed": parsed,
                    "comment": comment,
                    "query": query,
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": message,
                })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": parsed.get("description", "リクエストを解釈できませんでした"),
            })


def main():
    """メイン関数"""
    init_session_state()

    # タイトル
    st.title("🏀 NBA Player Analytics")
    st.markdown("NBA選手をいろんな角度で分析できます。自然言語で好きな分析をしてみてください！")
    st.caption(
        "💡 LLMはClaude Haiku 4.5を使用しています。"
        "選手名が正しい日本語表記にならないことがあります。"
        "APIエラーや分析エラーが発生することもありますが、ご容赦ください🙏"
    )

    # サイドバー
    render_sidebar()

    # チャット履歴を表示
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🏀"):
            st.write(msg["content"])

            # 結果がある場合は表示
            if "result" in msg:
                render_result(msg["result"], msg.get("parsed", {}), idx, msg.get("comment", ""), msg.get("query", ""))

    # サイドバーの例からのクエリをチェック
    if "pending_query" in st.session_state:
        query = st.session_state.pending_query
        del st.session_state.pending_query
        process_query(query)
        st.rerun()

    # チャット入力（Enterで送信）
    if prompt := st.chat_input("分析したいことを入力（例: コービー対レブロンのデュエル）"):
        process_query(prompt.strip())
        st.rerun()


if __name__ == "__main__":
    main()

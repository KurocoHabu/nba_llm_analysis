"""NBAオレンジテーマ カスタムCSS"""

# カラーパレット
COLORS = {
    "nba_orange": "#F26522",
    "nba_blue": "#1D428A",
    "bg_dark": "#0E1117",
    "bg_card": "#1E2530",
    "bg_hover": "#2A3441",
    "text_primary": "#FFFFFF",
    "text_secondary": "#B0B8C1",
    "accent_red": "#C8102E",
    "accent_gold": "#FDB927",
    "chart_bar": "#F26522",
    "chart_grid": "#3A4555",
}

# Streamlit用カスタムCSS
CUSTOM_CSS = """
<style>
/* 全体の背景 */
.stApp {
    background-color: #0E1117;
}

/* ヘッダー */
.stApp header {
    background-color: #0E1117;
}

/* サイドバー */
[data-testid="stSidebar"] {
    background-color: #1E2530;
}

[data-testid="stSidebar"] .stMarkdown {
    color: #FFFFFF;
}

/* チャット入力 */
.stChatInput {
    border-color: #F26522 !important;
}

.stChatInput:focus-within {
    border-color: #F26522 !important;
    box-shadow: 0 0 0 1px #F26522 !important;
}

/* チャットメッセージ - ユーザー */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #1D428A;
    border-radius: 12px;
    margin: 8px 0;
}

/* チャットメッセージ - アシスタント */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #1E2530;
    border-radius: 12px;
    margin: 8px 0;
}

/* ボタン - プライマリ */
.stButton > button {
    background-color: #F26522;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #FF7A3D;
    border: none;
}

/* ダウンロードボタン */
.stDownloadButton > button {
    background-color: #F26522;
    color: white;
    border: none;
    border-radius: 8px;
}

.stDownloadButton > button:hover {
    background-color: #FF7A3D;
}

/* タブ */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #1E2530;
    color: #B0B8C1;
    border-radius: 8px 8px 0 0;
    padding: 8px 16px;
}

.stTabs [aria-selected="true"] {
    background-color: #2A3441;
    color: #FFFFFF;
    border-bottom: 2px solid #F26522;
}

/* データフレーム */
.stDataFrame {
    background-color: #1E2530;
}

/* スピナー */
.stSpinner > div {
    border-top-color: #F26522 !important;
}

/* エラーメッセージ */
.stAlert {
    background-color: rgba(200, 16, 46, 0.2);
    border-color: #C8102E;
}

/* 成功メッセージ */
.stSuccess {
    background-color: rgba(242, 101, 34, 0.2);
    border-color: #F26522;
}

/* タイトル */
h1 {
    color: #F26522 !important;
}

/* サブタイトル */
h2, h3 {
    color: #FFFFFF !important;
}

/* 通常テキスト - 白ベースで視認性向上 */
p, span, label {
    color: #FFFFFF !important;
}

/* チャットメッセージ内のテキスト */
[data-testid="stChatMessage"] p {
    color: #FFFFFF !important;
}

/* マークダウンテキスト */
.stMarkdown {
    color: #FFFFFF !important;
}

/* コード */
code {
    background-color: #2A3441 !important;
    color: #FDB927 !important;
    padding: 4px 8px !important;
    border-radius: 4px !important;
}

/* stCodeブロック */
[data-testid="stCode"] {
    background-color: #2A3441 !important;
}

[data-testid="stCode"] code {
    color: #FDB927 !important;
    background-color: transparent !important;
}

[data-testid="stCode"] pre {
    background-color: #2A3441 !important;
    color: #FDB927 !important;
}

/* エクスパンダー */
.streamlit-expanderHeader {
    background-color: #1E2530;
    color: #FFFFFF;
}

/* メトリクス */
[data-testid="stMetricValue"] {
    color: #F26522;
}

/* セレクトボックス */
.stSelectbox > div > div {
    background-color: #1E2530;
    color: #FFFFFF;
}
</style>
"""


def get_plotly_theme():
    """Plotlyグラフ用のテーマ設定を返す"""
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": COLORS["text_primary"]},
        "xaxis": {
            "gridcolor": COLORS["chart_grid"],
            "zerolinecolor": COLORS["chart_grid"],
        },
        "yaxis": {
            "gridcolor": COLORS["chart_grid"],
            "zerolinecolor": COLORS["chart_grid"],
        },
    }


def get_bar_color():
    """棒グラフの色を返す"""
    return COLORS["chart_bar"]

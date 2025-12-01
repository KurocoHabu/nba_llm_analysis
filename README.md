# NBA Player Analytics

NBAの選手スタッツを自然言語で分析できるStreamlitアプリケーションです。

## 機能

- 自然言語でNBAスタッツを分析
- 各種ランキング（通算得点、連続記録、到達試合数など）
- 選手デュエル（直接対決）分析
- インタラクティブなグラフ表示
- CSV出力・Xシェア機能

## セットアップ

### 1. 仮想環境の作成

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 3. 環境変数の設定

`.streamlit/secrets.toml`にClaude APIキーを設定：

```toml
ANTHROPIC_API_KEY = "your-api-key-here"
```

### 4. アプリの起動

```bash
streamlit run app/main.py
```

## プロジェクト構成

```
nba_llm_analysis_app/
├── app/                    # Streamlitアプリ
│   ├── main.py            # メインエントリポイント
│   ├── prompts.py         # LLMプロンプト定義
│   ├── llm_interpreter.py # LLM連携
│   ├── executor.py        # 分析実行
│   ├── styles.py          # UIスタイル
│   └── query_history.py   # 履歴管理
├── src/                    # 分析ロジック
│   ├── analysis.py        # NBAAnalyzerクラス
│   ├── data_loader.py     # データ読み込み
│   └── utils.py           # ユーティリティ
├── data/                   # データファイル
│   ├── boxscore1946-2025.csv
│   ├── games1946-2025.csv
│   ├── Players_data_Latest.csv
│   └── player_imageURL.csv
├── .streamlit/             # Streamlit設定
└── requirements.txt        # 依存パッケージ
```

## 使い方

質問を入力すると、NBAスタッツを分析して結果を表示します。

### 質問例

- 25歳時点での通算得点ランキング
- 連続ダブルダブル記録TOP20
- 連勝記録ランキング
- 1万得点到達までの試合数TOP15
- プレイオフでの40得点ゲーム回数ランキング
- コービー対レブロンのデュエル

### 対応スタッツ

`PTS, TRB, AST, STL, BLK, 3P, Win, DD, TD`

### 試合タイプ

- `regular`: レギュラーシーズン
- `playoff`: プレイオフ
- `final`: ファイナル
- `all`: 全試合

## 技術スタック

- **Frontend**: Streamlit
- **LLM**: Claude (Anthropic API)
- **Visualization**: Plotly
- **Data Processing**: Pandas

## ライセンス

MIT License

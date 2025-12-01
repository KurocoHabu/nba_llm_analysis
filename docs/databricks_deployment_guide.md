# Databricks デプロイガイド

このガイドでは、NBA LLM Analysis AppをDatabricksにデプロイする方法を説明します。

## 目次

1. [前提条件](#前提条件)
2. [方法1: GUI（手動アップロード）](#方法1-gui手動アップロード)
3. [方法2: Databricks CLI](#方法2-databricks-cli)
4. [方法3: Claude Code + MCP連携](#方法3-claude-code--mcp連携)
5. [Streamlitアプリの起動](#streamlitアプリの起動)
6. [トラブルシューティング](#トラブルシューティング)

---

## 前提条件

- Databricksアカウント
- ワークスペースへのアクセス権限
- Unity Catalogが有効（推奨）

---

## 方法1: GUI（手動アップロード）

最も簡単な方法です。Databricksの管理画面から直接操作します。

### Step 1: データファイルのアップロード

1. **Databricksワークスペース**にログイン
2. 左サイドバーから **「Catalog」** をクリック
3. **「+ Add」→「Add data」** をクリック
4. **「Upload files」** を選択
5. 以下のファイルをアップロード:
   - `data/boxscore1946-2025.csv`
   - `data/games1946-2025.csv`
   - `data/Players_data_Latest.csv`
   - `data/player_imageURL.csv`
   - `data/team_images.csv`

6. アップロード先を指定:
   - Catalog: `main`（または任意）
   - Schema: `nba_analysis`（新規作成）
   - Table名: ファイル名に基づいて自動設定

### Step 2: コードのアップロード（Repos経由）

1. 左サイドバーから **「Workspace」** をクリック
2. **「Repos」** フォルダを選択
3. **「Add」→「Git folder」** をクリック
4. GitHubリポジトリURLを入力:
   ```
   https://github.com/nba-bcr/nba_llm_analysis.git
   ```
5. **「Create Git folder」** をクリック

### Step 3: Secretsの設定（APIキー）

1. 左サイドバーから **「Compute」** をクリック
2. 使用するクラスターを選択
3. **「Apps」→「New App」** で新規Appを作成する場合:
   - App作成後、**「Settings」→「Secrets」** でAPIキーを追加

   または、**Databricks CLI**でSecret Scopeを作成:
   ```bash
   databricks secrets create-scope nba-analysis
   databricks secrets put-secret nba-analysis ANTHROPIC_API_KEY
   ```

### Step 4: Databricks Appの作成

1. 左サイドバーから **「Compute」** をクリック
2. **「Apps」** タブを選択
3. **「Create app」** をクリック
4. 以下を設定:
   - **Name**: `nba-llm-analysis`
   - **Source**: Repos（Step 2で追加したリポジトリ）
   - **Path**: `/app/main.py`
   - **Framework**: Streamlit
5. **「Create」** をクリック
6. アプリがデプロイされるまで待機（数分）

---

## 方法2: Databricks CLI

コマンドラインから操作する方法です。

### Step 1: CLIの設定

```bash
# 認証設定（トークン方式）
databricks configure --token

# 以下を入力:
# - Databricks Host: https://your-workspace.cloud.databricks.com
# - Token: dapi... (Personal Access Token)
```

**Personal Access Tokenの取得方法:**
1. Databricksワークスペースにログイン
2. 右上のユーザーアイコン → **「Settings」**
3. **「Developer」→「Access tokens」**
4. **「Generate new token」** をクリック

### Step 2: ファイルのアップロード

```bash
# DBFSにデータディレクトリを作成
databricks fs mkdirs dbfs:/FileStore/nba_analysis/data

# データファイルをアップロード
databricks fs cp data/boxscore1946-2025.csv dbfs:/FileStore/nba_analysis/data/
databricks fs cp data/games1946-2025.csv dbfs:/FileStore/nba_analysis/data/
databricks fs cp data/Players_data_Latest.csv dbfs:/FileStore/nba_analysis/data/
databricks fs cp data/player_imageURL.csv dbfs:/FileStore/nba_analysis/data/
databricks fs cp data/team_images.csv dbfs:/FileStore/nba_analysis/data/
```

### Step 3: コードの配置

```bash
# アプリコードをアップロード
databricks fs mkdirs dbfs:/FileStore/nba_analysis/app
databricks fs cp -r app/ dbfs:/FileStore/nba_analysis/app/
databricks fs cp -r src/ dbfs:/FileStore/nba_analysis/src/
databricks fs cp requirements.txt dbfs:/FileStore/nba_analysis/
```

---

## 方法3: Claude Code + MCP連携

Claude Codeから直接Databricksを操作する方法です。

### Step 1: Databricks MCPサーバーの追加

```bash
# Databricks Managed MCPを追加
claude mcp add --transport http databricks-server \
  https://your-workspace.cloud.databricks.com/api/2.0/mcp/gold/core
```

または、コミュニティ版のMCPサーバーを使用:

```bash
# databricks-mcp をインストール
pip install databricks-mcp

# MCPサーバーを追加
claude mcp add-json "databricks" '{
  "command": "python",
  "args": ["-m", "databricks_mcp"],
  "env": {
    "DATABRICKS_HOST": "https://your-workspace.cloud.databricks.com",
    "DATABRICKS_TOKEN": "dapi..."
  }
}'
```

### Step 2: Claude Codeから操作

MCPが設定されると、Claude Codeから以下のような操作が可能になります:

- クラスターの管理
- ファイルのアップロード/ダウンロード
- SQLの実行
- ノートブックの操作

---

## Streamlitアプリの起動

### Databricks Appsを使用する場合

1. **「Compute」→「Apps」** から作成したアプリを選択
2. **「Open app」** をクリック
3. アプリのURLが発行される

### ノートブックから起動する場合

Databricksノートブックで以下を実行:

```python
# データパスを更新（Databricks用）
import os
os.environ["DATA_DIR"] = "/dbfs/FileStore/nba_analysis/data"

# Streamlitを起動（ノートブックでは制限あり）
# Databricks Appsの使用を推奨
```

---

## コードの修正（必要な場合）

Databricks環境でデータを読み込むために、`src/data_loader.py`を修正する必要があるかもしれません:

```python
# 変更前
def __init__(self, data_dir: str = "data"):

# 変更後（環境変数で切り替え）
import os
def __init__(self, data_dir: str = None):
    if data_dir is None:
        data_dir = os.environ.get("DATA_DIR", "data")
    self.data_dir = Path(data_dir)
```

---

## トラブルシューティング

### 「Permission denied」エラー
- Unity Catalogの権限を確認
- クラスターがデータにアクセスできるか確認

### 「File not found」エラー
- DBFSパスが正しいか確認（`/dbfs/`プレフィックス）
- ファイルが正しくアップロードされているか確認:
  ```bash
  databricks fs ls dbfs:/FileStore/nba_analysis/data/
  ```

### APIキーエラー
- Secret Scopeが正しく設定されているか確認
- アプリのEnvironment Variablesを確認

### Streamlitが起動しない
- `requirements.txt`の依存関係を確認
- Pythonバージョンの互換性を確認（3.9以上推奨）

---

## 参考リンク

- [Databricks Apps Documentation](https://docs.databricks.com/en/dev-tools/databricks-apps/index.html)
- [Databricks MCP Servers](https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp)
- [Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/index.html)

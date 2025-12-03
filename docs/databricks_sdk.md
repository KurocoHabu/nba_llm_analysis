# Databricks SDK について

## SDKとは

SDK (Software Development Kit) は、特定のプラットフォームやサービスを操作するためのプログラミングツールキットです。

**Databricks SDK** は、Pythonコードから Databricks の機能にアクセスするためのライブラリです。

## このアプリでの使用箇所

### 1. Volumes からのファイル読み込み

Databricks Apps では `/Volumes/...` パスに直接アクセスできないため、SDK を使ってファイルをダウンロードします。

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
response = w.files.download("/Volumes/workspace/nba_analysis/data/boxscore.csv")
file_content = response.contents.read()
```

**ファイル:** `src/data_loader.py` の `download_from_volume()` 関数

### 2. Secrets からの API キー取得

Anthropic API キーを安全に保存・取得するために使用します。

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
response = w.secrets.get_secret(scope="nba-app", key="ANTHROPIC_API_KEY")
api_key = response.value
```

**ファイル:** `app/llm_interpreter.py` の `_get_api_key_from_databricks()` 関数

## なぜ SDK が必要か

| 環境 | 直接パスアクセス | SDK経由 |
|------|------------------|---------|
| ローカル | ✅ 可能 | 不要 |
| Streamlit Cloud | ✅ 可能 | 不要 |
| **Databricks Apps** | ❌ 不可 | **必要** |

Databricks Apps はセキュリティ上の理由から、Volume やSecrets に直接アクセスできません。SDK を経由することで、適切な認証・認可を行ってアクセスします。

## インストール

```bash
pip install databricks-sdk
```

`requirements.txt` に以下が含まれています：
```
databricks-sdk>=0.20.0
```

## 認証

Databricks Apps 環境では、アプリに割り当てられたサービスプリンシパルが自動的に認証を行うため、明示的な認証設定は不要です。

ローカル開発時は `databricks configure` で設定した認証情報が使用されます。

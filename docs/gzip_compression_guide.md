# gzip圧縮ガイド

## gzipとは？

gzip（GNU zip）は、ファイルを圧縮するための標準的なアルゴリズムです。特にテキストベースのファイル（CSV、JSON、ログファイルなど）に対して高い圧縮率を発揮します。

## なぜ圧縮が必要か？

### 今回の問題
- `boxscore1946-2025.csv`: **185MB**
- GitHubのファイルサイズ制限: **100MB**
- Git LFSを使用 → Streamlit CloudがGit LFSをサポートしていない

### 解決策
- gzip圧縮で **185MB → 約20-30MB** に削減
- Git LFSなしで通常のgitでコミット可能
- Streamlit Cloudで問題なくデプロイ可能

## 圧縮率の目安

| ファイルタイプ | 圧縮率 | 例 |
|--------------|--------|-----|
| CSV（テキスト） | 80-90% | 185MB → 20-30MB |
| JSON | 80-90% | 100MB → 10-20MB |
| 画像（PNG/JPG） | 0-10% | すでに圧縮済み |
| バイナリ | 20-50% | 様々 |

CSVファイルは繰り返しパターンが多いため、非常に高い圧縮率が得られます。

## 使い方

### コマンドラインで圧縮

```bash
# 圧縮（元ファイルは残る）
gzip -k data/boxscore1946-2025.csv

# 結果: data/boxscore1946-2025.csv.gz が作成される
```

### コマンドラインで解凍

```bash
# 解凍
gunzip data/boxscore1946-2025.csv.gz

# または
gzip -d data/boxscore1946-2025.csv.gz
```

### Pythonで圧縮ファイルを読み込む

pandasは`.gz`ファイルを自動的に認識して読み込めます：

```python
import pandas as pd

# 通常のCSV読み込み
df = pd.read_csv("data/boxscore1946-2025.csv")

# gzip圧縮CSVの読み込み（同じ書き方でOK）
df = pd.read_csv("data/boxscore1946-2025.csv.gz")

# 明示的に指定する場合
df = pd.read_csv("data/boxscore1946-2025.csv.gz", compression="gzip")
```

### Pythonで圧縮して保存

```python
import pandas as pd

df = pd.read_csv("data/boxscore1946-2025.csv")

# gzip圧縮して保存
df.to_csv("data/boxscore1946-2025.csv.gz", index=False, compression="gzip")
```

## 今回の実装手順

### 1. CSVファイルを圧縮

```bash
cd /path/to/nba_llm_analysis_app

# 大きいファイルを圧縮
gzip -k data/boxscore1946-2025.csv
gzip -k data/games1946-2025.csv
```

### 2. Git LFSから除外

```bash
# .gitattributesからLFS設定を削除
# または、圧縮ファイルは通常のgitで管理

# 元の大きいファイルをgitignoreに追加
echo "data/boxscore1946-2025.csv" >> .gitignore
echo "data/games1946-2025.csv" >> .gitignore
```

### 3. コードを修正

`src/data_loader.py`でファイル名を変更：

```python
# 変更前
def load_boxscore(self, filename: str = "boxscore1946-2025.csv"):

# 変更後
def load_boxscore(self, filename: str = "boxscore1946-2025.csv.gz"):
```

### 4. コミット＆プッシュ

```bash
git add data/*.gz
git add src/data_loader.py
git commit -m "Use gzip compressed CSV files"
git push
```

### 5. Streamlit Cloudで再デプロイ

Streamlit Cloudの管理画面でRebootすれば、新しいコードとデータが反映されます。

## メリット・デメリット

### メリット
- ファイルサイズが大幅に削減
- Git LFSが不要になる
- デプロイが高速化
- ストレージコストの削減

### デメリット
- 読み込み時にCPUを使用（解凍処理）
- ただし、pandasは最適化されているため、ほとんど気にならない
- 初回読み込みが若干遅くなる可能性（数秒程度）

## 圧縮レベルの調整

gzipには1-9の圧縮レベルがあります：

```bash
# 高速だが圧縮率低い
gzip -1 data/boxscore1946-2025.csv

# バランス（デフォルト）
gzip -6 data/boxscore1946-2025.csv

# 最高圧縮率だが遅い
gzip -9 data/boxscore1946-2025.csv
```

通常はデフォルト（-6）で十分です。

## 参考リンク

- [pandas read_csv compression](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
- [gzip man page](https://www.gnu.org/software/gzip/manual/gzip.html)
- [GitHub file size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

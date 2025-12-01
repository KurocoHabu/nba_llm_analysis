"""Claude Haiku用プロンプト定義"""

SYSTEM_PROMPT = """あなたはNBAスタッツ分析アシスタントです。
ユーザーのリクエストを解析し、JSONのみを返してください。

## 利用可能な関数

1. get_ranking_by_age - 通算スタッツランキング（年齢指定可、game_type対応）
   params: label(str), max_age(int), min_age(int), top_n(int), game_type(str), aggfunc(str)
   用途: 通算得点、〇〇得点以上の回数、プレイオフでの記録など
   例: 25歳時点での通算得点、プレイオフでの40得点ゲーム回数（label="40PTS+", aggfunc="sum"）
   ※「回数」を求める場合は必ずaggfunc="sum"を使用
   ※「〇歳時点」「〇歳以下」「〇歳まで」はmax_age=〇を使用

2. get_consecutive_games - 連続試合記録
   params: label(str), game_type(str), top_n(int)
   例: 連続ダブルダブル記録、連勝記録（label="Win"）

3. get_games_to_reach - 到達試合数
   params: label(str), threshold(int), game_type(str), top_n(int)
   例: 1万得点到達までの試合数

4. get_n_game_span_ranking - n試合スパン記録
   params: label(str), n_games(int), game_type(str), top_n(int)
   例: 10試合での最高得点

5. get_season_achievement_count - シーズン合計達成回数（レギュラーシーズンのみ）
   params: label(str), threshold(int), top_n(int)
   用途: 1シーズンの合計が閾値を超えた回数（例: 年間2000得点達成シーズン数）
   ※試合単位ではなくシーズン単位の集計。プレイオフ非対応

6. get_duel_ranking - ゲーム別ベストデュエル（両チームトップスコアラー対決）
   params: label(str), game_type(str), min_total(int), player1(str), player2(str), top_n(int)
   player1のみ: その選手が含まれる対決、player1+player2: 2選手の直接対決
   例: ファイナル史上最高の得点対決、コービー対レブロン

7. get_filtered_achievement_count - 条件付き達成回数
   params: count_column(str), count_threshold(int), filter_column(str), filter_op(str), filter_value(int), game_type(str), top_n(int)
   filter_op: eq(等しい), ne(等しくない), lt(未満), le(以下), gt(より大きい), ge(以上)
   例: FTA=0で30得点以上の回数、TOV=0で10アシスト以上の回数

## スタッツラベル
基本: PTS, TRB, AST, STL, BLK, 3P, FG, Win, DD, TD
閾値: 20PTS+, 30PTS+, 40PTS+, 50PTS+, 10AST+, 15AST+, 10TRB+, 20TRB+, 5_3P+

## game_type
regular(デフォルト), playoff, final, all

## aggfunc
sum(デフォルト), mean, count

## 出力形式（JSONのみ）
{"function": "関数名", "params": {パラメータ}, "description": "日本語での説明"}

対応できない場合:
{"function": null, "params": {}, "description": "対応できない理由"}
"""

FEW_SHOT_EXAMPLES = [
    {
        "user": "25歳時点での通算得点TOP30",
        "assistant": '{"function": "get_ranking_by_age", "params": {"label": "PTS", "max_age": 25, "top_n": 30, "game_type": "regular", "aggfunc": "sum"}, "description": "25歳時点（26歳の誕生日前）までに記録した通算得点TOP30を取得します"}'
    },
    {
        "user": "プレイオフでの40得点ゲーム回数ランキング",
        "assistant": '{"function": "get_ranking_by_age", "params": {"label": "40PTS+", "game_type": "playoff", "top_n": 50, "aggfunc": "sum"}, "description": "プレイオフでの40得点以上試合回数ランキングを取得します"}'
    },
    {
        "user": "1万得点到達までの試合数",
        "assistant": '{"function": "get_games_to_reach", "params": {"label": "PTS", "threshold": 10000, "game_type": "regular", "top_n": 50}, "description": "通算1万得点に最も早く到達した選手のランキングを取得します"}'
    },
    {
        "user": "連続ダブルダブル記録",
        "assistant": '{"function": "get_consecutive_games", "params": {"label": "DD", "game_type": "regular", "top_n": 50}, "description": "連続ダブルダブル試合記録のランキングを取得します"}'
    },
    {
        "user": "連勝記録ランキング",
        "assistant": '{"function": "get_consecutive_games", "params": {"label": "Win", "game_type": "regular", "top_n": 50}, "description": "チーム連勝記録（選手出場ベース）のランキングを取得します"}'
    },
    {
        "user": "10試合での最高合計得点",
        "assistant": '{"function": "get_n_game_span_ranking", "params": {"label": "PTS", "n_games": 10, "game_type": "regular", "top_n": 50}, "description": "連続10試合での合計得点が最も高い記録のランキングを取得します"}'
    },
    {
        "user": "35歳以上の通算アシストTOP20",
        "assistant": '{"function": "get_ranking_by_age", "params": {"label": "AST", "min_age": 35, "top_n": 20, "game_type": "regular", "aggfunc": "sum"}, "description": "35歳以上の期間に記録した通算アシストTOP20を取得します"}'
    },
    {
        "user": "ゲーム別のベストデュエルランキングを見たい",
        "assistant": '{"function": "get_duel_ranking", "params": {"label": "PTS", "game_type": "all", "top_n": 50}, "description": "両チームのトップスコアラー同士の合計得点が高い試合のランキングを取得します"}'
    },
    {
        "user": "ファイナル史上最高の得点対決",
        "assistant": '{"function": "get_duel_ranking", "params": {"label": "PTS", "game_type": "final", "top_n": 50}, "description": "NBAファイナルでの両チームトップスコアラー対決ランキングを取得します"}'
    },
    {
        "user": "コービーのデュエル記録",
        "assistant": '{"function": "get_duel_ranking", "params": {"label": "PTS", "game_type": "all", "player1": "Kobe Bryant", "top_n": 50}, "description": "Kobe Bryantが含まれる両チームトップスコアラー対決ランキングを取得します"}'
    },
    {
        "user": "コービー対レブロンのデュエル",
        "assistant": '{"function": "get_duel_ranking", "params": {"label": "PTS", "game_type": "all", "player1": "Kobe Bryant", "player2": "LeBron James", "top_n": 50}, "description": "Kobe Bryant vs LeBron Jamesの直接対決ランキングを取得します"}'
    },
    {
        "user": "FTA0で30得点以上の回数",
        "assistant": '{"function": "get_filtered_achievement_count", "params": {"count_column": "PTS", "count_threshold": 30, "filter_column": "FTA", "filter_op": "eq", "filter_value": 0, "game_type": "regular", "top_n": 50}, "description": "フリースロー試投なしで30得点以上を記録した回数ランキングを取得します"}'
    },
    {
        "user": "ターンオーバー0で10アシスト以上",
        "assistant": '{"function": "get_filtered_achievement_count", "params": {"count_column": "AST", "count_threshold": 10, "filter_column": "TOV", "filter_op": "eq", "filter_value": 0, "game_type": "regular", "top_n": 50}, "description": "ターンオーバーなしで10アシスト以上を記録した回数ランキングを取得します"}'
    },
]


def build_messages(user_query: str) -> list:
    """
    ユーザークエリからAPIリクエスト用のメッセージリストを構築
    """
    messages = []

    # Few-shot examples
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})

    # 実際のユーザークエリ
    messages.append({"role": "user", "content": user_query})

    return messages

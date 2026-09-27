# text-mining — Claude Context

日本語テキストマイニング用の Streamlit アプリ（形態素解析・可視化）。

## スタック

- Python / Streamlit / GiNZA (ja_ginza) / scikit-learn / matplotlib / altair
- venv: `.venv/`（gitignore対象）
- 現在の本体: `src/text_mining_app.py`（`_attic/` 配下の app_v100.py 等は旧版・参照専用、編集しない）

## モジュール構成（`src/`）

解析ロジックは `src/core/` に分離されている。「本体は text_mining_app.py 一枚だけ」ではない。

| ファイル | 役割 |
|---|---|
| `text_mining_app.py` | Streamlit UI 本体。タブ構成・データ入力・画面遷移 |
| `run.py` | exe 起動用エントリーポイント（PyInstaller 専用）。8501 から空きポートを探して待受し、`localhost` をブラウザで開く |
| `core/config.py` | `text_mining_config.json` の load/save、`resource_path()`（`sys._MEIPASS` 解決）、セッション初期化、フォント候補の検出 |
| `core/nlp_engine.py` | GiNZA モデル読込（`load_ginza_model`）・感情極性辞書のロード（`load_sentiment_dict`） |
| `core/stats.py` | 解析ロジック本体。`run_nlp_morphology` / `perform_stats_analysis`（TF-IDF・N-gram・共起・感情スコア）/ `perform_correspondence_analysis` / `export_analysis_to_excel` |
| `core/visualizer.py` | 描画。WordCloud・共起ネットワーク・対応分析プロット |

## 実行

```bash
scripts\text_mining_app_launch.bat
# 中身: .venv を有効化 → streamlit run src\text_mining_app.py
```

## テスト

```bash
PYTHONPATH=. pytest tests/ -v
```

CI（`.github/workflows/tests.yml`）準拠のコマンド。プロジェクトルートに `conftest.py`/`pytest.ini`/`pyproject.toml` が無いため、`PYTHONPATH=.` を付けずに素の `pytest` を実行すると `src.core...` 系のimportが失敗することがある。

pytest は `requirements.txt` に含まれているため、`pip install -r requirements.txt` で入る。

## 依存関係

- `requirements.txt` — アプリ実行・テスト用のみ
- `requirements-build.txt` — Windows exe ビルド専用（`pyinstaller`・`pefile`・`pywin32-ctypes` 等）。`-r requirements.txt` を含むので、ビルド環境はこちらだけ入れればよい
- `scripts/create_requirements.bat` — `.venv` の `pip freeze` からビルド専用パッケージを除外して `requirements.txt` を再生成する。ビルド専用パッケージを増やす場合は、このバッチの `findstr` フィルタと `requirements-build.txt` の両方を更新する。なお `( ... ) > requirements.txt` ブロック内の `echo` 行に**丸括弧を書くとブロックが早期終了してリダイレクトが壊れる**（無言で標準出力に漏れるだけなので気づきにくい）
- **`requirements.txt` のピンは CI の Python（3.11 / 3.12）で解決できる必要がある**。ローカルの `.venv` は 3.13 なので、freeze 由来のピンがそのまま CI で入るとは限らない（実際 `networkx==3.7` は `Requires-Python >=3.12` で 3.11 のジョブを落とした）。再生成後は最低でも以下で 3.11 を通すこと:
  ```bash
  pip install --dry-run --ignore-installed --python-version 3.11 --only-binary=:all: --target <任意の空ディレクトリ> -r requirements.txt
  ```

## 注意点

- `_attic/` は過去バージョンのアーカイブ。新機能はここに書かず `src/` 側で作業する
- `assets/NotoSansJP-Regular.ttf` は日本語グラフ描画用フォント
- 感情極性辞書（`assets/sentiment_dict.csv`）が無くても**アプリは起動する**。`load_sentiment_dict()` が空dictを返し、`stats.py` 側で感情スコア `0.0`・分類「ニュートラル」にフォールバックする（感情分析タブは全件ニュートラル表示になる）。辞書の取得は `python scripts/download_sentiment.py`（外部ネットワーク依存。CI は `|| true` で失敗を許容している）
- exe ビルド（`TextMiningApp.spec` / `scripts/build.bat`）は**ローカルまたはCI専用**。通常のアプリ改修では触らない。ビルドには `pip install -r requirements-build.txt` が必要で、成果物は `dist/`・`build/`（ともにgitignore対象）に出る

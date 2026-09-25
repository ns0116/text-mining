# AUDIT.md — text-mining

作成日: 2026-09-24
更新日: 2026-09-25（完了状況を整理）

## 完了状況（最終確認: 2026-09-25）

> 状態はこの表が正。下の監査本文は 2026-09-24 監査時点の記録（原文のまま）。

**総合: 🟡 高は完了 / 残り 5件（中3・低2）**

| 優先度 | 完了 | 残り |
|---|---|---|
| 高 | 2/2 | 0 |
| 中 | 0/3 | 3 |
| 低 | 0/2 | 2 |

| # | 優先度 | 項目 | 状態 | 備考 |
|---|---|---|---|---|
| 1 | 高 | CLAUDE.mdの.gitignore除外を見直す | ✅ 2026-09-24 | `5de4fb0`。機密情報・ローカル絶対パス・NAS/IP等なしを確認のうえ追跡対象化。代わりに`CLAUDE.local.md`を除外 |
| 2 | 高 | CLAUDE.mdにテスト実行手順を追記 | ✅ 2026-09-24 | `7f77934`。`PYTHONPATH=. pytest tests/ -v`（CI準拠） |
| 3 | 中 | 感情辞書DL失敗時の挙動をCLAUDE.md/READMEに明記 | ⬜ 未対応 | README/CLAUDE.mdとも失敗時フォールバックの記載なし（2026-09-25確認） |
| 4 | 中 | requirements.txtからビルド専用パッケージを分離 | ⬜ 未対応 | `pyinstaller`/`pefile`/`pywin32-ctypes`等が依然混在 |
| 5 | 中 | CLAUDE.mdに`src/core/`のモジュール構成を追記 | ⬜ 未対応 | |
| 6 | 低 | 感情辞書DL URLのHTTPS化可否を確認 | ⬜ 未対応 | `download_sentiment.py`は`http://`のまま。提供元のHTTPS対応確認が必要 |
| 7 | 低 | exeビルドはローカル/CI専用とCLAUDE.mdに注記 | ⬜ 未対応 | |

## プロンプト監査結果

### 調査対象の存在状況
- `CLAUDE.md`（プロジェクトルート）: **存在する**（22行、簡潔）
- `.claude/agents/*`: **対象ファイルなし**
- `.claude/skills/*`: **対象ファイルなし**
- `.claude/commands/*`: **対象ファイルなし**
- `.claude/rules/*`: **対象ファイルなし**
- 参考: `.claude/settings.local.json` は存在するが（`Bash(git fetch *)` / `Bash(git stash *)` の許可のみ）、git未追跡でありタスク1の対象外。指示内容としての問題はなし。

### 最重要の発見: CLAUDE.md自体がgitignore対象で非追跡
`.gitignore:21` に `CLAUDE.md` が登録されており（コミット `4ebddfa chore: CLAUDE.mdを.gitignoreに追加`, 2026-09-11）、`git ls-files` にも `git log -- CLAUDE.md` にも一切現れない＝**このファイルはリポジトリに一度もコミットされていない、完全にローカル限定のファイル**。
→ クラウドサンドボックス（claude.ai上のClaude Code on the web等）が`git clone`する対象には含まれないため、スタック情報・起動コマンド・`_attic/`の扱いといったCLAUDE.mdの内容はクラウドセッションから一切参照できない。プロンプト監査というよりクラウドセッション対応上の根本課題であり、タスク2の改善提案（優先度: 高）にも計上した。

### 古くなった情報・矛盾
- 特になし。CLAUDE.md内で言及されているファイル（`src/text_mining_app.py`、`_attic/app_v100.py`、`scripts/text_mining_app_launch.bat`、`assets/NotoSansJP-Regular.ttf`）はすべて実在し、記述と実態に矛盾はない。

### 曖昧・解釈がブレそうな箇所
- 「現在の本体: `src/text_mining_app.py`」という一文が、`src/core/`配下のモジュール（`config.py`, `nlp_engine.py`, `stats.py`(465行), `visualizer.py`(196行)）の存在に触れていないため、「本体はtext_mining_app.py一枚だけ」という誤解を招きうる。実際にはTF-IDF・N-gram・共起・対応分析などの中核ロジックは`src/core/stats.py`に、可視化は`visualizer.py`に分離されている。

### 冗長な記述
- 特になし。CLAUDE.mdは全体的に簡潔で、無駄なトークン消費は見られない。

### 抜けているコンテキスト（コード上は存在するがCLAUDE.mdに未記載）
1. **テスト実行コマンド**: `tests/`配下にpytestテストがあり`.github/workflows/tests.yml:29`で`PYTHONPATH=. pytest tests/ -v`として実行しているが、CLAUDE.mdにはテストの実行方法が一切書かれていない。README.mdには`pytest`とだけ案内されており（`PYTHONPATH=.`の言及なし）、プロジェクトルートに`conftest.py`/`pytest.ini`/`pyproject.toml`が存在しないため、素の`pytest`実行では`from src.core...`のimportが失敗する可能性がある。CLAUDE.mdにCI準拠のコマンドを明記すべき。
2. **感情分析辞書のセットアップ**: `scripts/download_sentiment.py`で外部サイトから辞書をダウンロードする手順が必要（README.mdには記載あるがCLAUDE.mdにはなし）。感情分析機能に触れる作業をする際に重要な前提情報が抜けている。
3. **ビルド手順（exe化）**: `TextMiningApp.spec` / `scripts/build.bat`によるPyInstallerビルドについて、CLAUDE.mdは一切触れていない。ローカル/CI専用の作業か、Claude Codeが触ってよい領域かの方針が書かれていない。

## ローカル依存リスト

1. **`.gitignore:21`（`CLAUDE.md`）** — CLAUDE.md自体がgit非追跡。クラウドセッションでは`git clone`後にこのファイルが存在しない（詳細は上記「最重要の発見」参照）。
2. **`src/core/config.py:41-52`** — `get_system_font_options()`内にWindows/Mac/Linuxのフォント絶対パスがハードコードされている（例: `C:/Windows/Fonts/YuGothM.ttc`, `/System/Library/Fonts/Hiragino Sans GB.ttc`, `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`）。ただし`os.path.exists`判定後に採用するフォールバック設計のため実害はなく、クラウド(Linux)サンドボックスでは自動的に添付の`Noto Sans JP`のみが候補になる（正常動作）。
3. **`scripts/download_sentiment.py:16-17`** — 東北大学 乾・関根研究室の感情極性辞書を`http://www.cl.ecei.tohoku.ac.jp/...`（平文HTTP・外部ネットワーク）から取得。クラウドサンドボックスのネットワークポリシー次第では到達できない可能性がある。CI(`.github/workflows/tests.yml:24-25`)では`|| true`でフォールバックしているが、失敗時にアプリがどう振る舞うか（辞書なしで感情分析機能が動くのか）がCLAUDE.md/READMEに明記されていない。
4. **`src/run.py:17,30`** — `localhost`固定でのポート待受・`webbrowser.open`呼び出し。PyInstaller化したexe専用のエントリーポイントで、GUIのないヘッドレスなクラウドサンドボックスでは`webbrowser.open`が意味をなさない可能性があるが、通常の`streamlit run src/text_mining_app.py`起動経路（README推奨手順）には影響しない。
5. **`requirements.txt:45,56,57,61`** — `pefile`, `pyinstaller`, `pyinstaller-hooks-contrib`, `pywin32-ctypes`などWindows exeビルド専用パッケージが本番実行用の依存と混在。CI(ubuntu-latest)で`pip install -r requirements.txt`は現に成功しており致命的ではないが、Streamlitアプリを動かすだけのクラウドセッションにとっては不要なインストール（GiNZAモデル等の大容量パッケージ群を含む）が発生し、セットアップが遅くなる。
6. **`TextMiningApp.spec`全体** — PyInstallerによるWindows/mac向けexeビルド設定。クラウドセッションの用途（コード編集・テスト）には無関係だが、その旨のドキュメントがない。

備考: NASのUNCパス（`\\...`）、WSL2の`/mnt/`、社内プライベートIPなど、プロジェクトのソースコード・設定ファイル上には見つからなかった（`.venv/`を除く全ファイルをgrep）。`localhost`/`127.0.0.1`への言及は`src/run.py`とREADMEのみで、いずれも実害は限定的。

## 改善提案（優先度付き）

### 高
- ✅ **対応済み** ~~CLAUDE.mdの.gitignore除外を見直す~~: `.gitignore`から`CLAUDE.md`を外し、Git追跡対象に変更した（2026-09-24）。
- ✅ **対応済み** ~~CLAUDE.mdにテスト実行手順を追記~~: `PYTHONPATH=. pytest tests/ -v`（CI準拠）をCLAUDE.mdに追記した（2026-09-24）。

### 中
- **感情辞書ダウンロード失敗時の挙動を明記**: ネットワーク制限があるクラウドサンドボックスでも`scripts/download_sentiment.py`が失敗した場合にアプリ本体は起動できること（`nlp_engine.py:24`の`os.path.exists`チェックで空辞書にグレースフルフォールバックする）をCLAUDE.md/READMEに記載する。
- **requirements.txtのビルド専用パッケージ分離**: `pyinstaller`, `pywin32-ctypes`, `pefile`等をWindows exeビルド専用の`requirements-build.txt`のような別ファイルに切り出し、アプリ実行のみが目的の環境（クラウドセッション含む）のセットアップを軽量化する。
- **CLAUDE.mdにsrc/core/配下のモジュール構成を追記**: `config.py`/`nlp_engine.py`/`stats.py`/`visualizer.py`の役割を一行ずつ要約し、「本体はtext_mining_app.pyのみ」という誤解を防ぐ。

### 低
- 感情辞書のダウンロードURLをHTTPS化できないか確認する（提供元サイト側の対応次第）。
- `TextMiningApp.spec`/`scripts/build.bat`によるexeビルドは「ローカル/CI専用作業であり、クラウドセッションでは通常触らない」旨をCLAUDE.mdに一言添える。

# AUDIT.md — text-mining

作成日: 2026-09-24
更新日: 2026-09-27（監査7件すべて対応・残課題2件を記録）

## 完了状況（最終確認: 2026-09-27）

> 状態はこの表が正。下の「監査本文」「ローカル依存リスト」は 2026-09-24 監査時点の記録（原文のまま）で、**対応済みの項目については現状と食い違う箇所がある**（例: ローカル依存リスト3の`http://`、同5のビルド専用パッケージ混在、同6のビルド手順未文書化はいずれも解決済み）。現状は上の表と下の「残課題」を参照。

**総合: 🟢 監査7件は全完了 / 残課題2件（下記）**

| 優先度 | 完了 | 残り |
|---|---|---|
| 高 | 2/2 | 0 |
| 中 | 3/3 | 0 |
| 低 | 2/2 | 0 |

| # | 優先度 | 項目 | 状態 | 備考 |
|---|---|---|---|---|
| 1 | 高 | CLAUDE.mdの.gitignore除外を見直す | ✅ 2026-09-24 | `5de4fb0`。機密情報・ローカル絶対パス・NAS/IP等なしを確認のうえ追跡対象化。代わりに`CLAUDE.local.md`を除外 |
| 2 | 高 | CLAUDE.mdにテスト実行手順を追記 | ✅ 2026-09-24 | `7f77934`。`PYTHONPATH=. pytest tests/ -v`（CI準拠） |
| 3 | 中 | 感情辞書DL失敗時の挙動をCLAUDE.md/READMEに明記 | ✅ 2026-09-27 | `ec4605b`。空辞書→感情スコア0.0・「ニュートラル」にフォールバックする挙動をCLAUDE.md注意点とREADME手順3に記載 |
| 4 | 中 | requirements.txtからビルド専用パッケージを分離 | ✅ 2026-09-27 | `ec4605b`。`requirements-build.txt`新設。`pyinstaller`/`pyinstaller-hooks-contrib`/`pefile`/`pywin32-ctypes`/`altgraph`を移動。`create_requirements.bat`に除外フィルタ、`build.bat`にインストール手順を追加 |
| 5 | 中 | CLAUDE.mdに`src/core/`のモジュール構成を追記 | ✅ 2026-09-27 | `ec4605b`。`## モジュール構成（src/）`を追加（config/nlp_engine/stats/visualizer/run.py） |
| 6 | 低 | 感情辞書DL URLのHTTPS化可否を確認 | ✅ 2026-09-27 | `ec4605b`。`https://www.cl.ecei.tohoku.ac.jp/...` で両URLとも200（バイト数はHTTPと同一）を確認し、`download_sentiment.py`をhttpsに変更 |
| 7 | 低 | exeビルドはローカル/CI専用とCLAUDE.mdに注記 | ✅ 2026-09-27 | `ec4605b`。CLAUDE.md注意点に、ビルドはローカル/CI専用・`requirements-build.txt`が必要・成果物は`dist/`/`build/`である旨を追記 |

## 残課題（2026-09-27時点・未対応）

監査7件は完了。以下は対応中に見つかった未解決項目で、いずれも**方針判断またはローカル環境の操作**が必要なため未着手。

| # | 項目 | 状態 | 内容 |
|---|---|---|---|
| A | CIの辞書依存テストがフレーキー | 方針判断待ち | `tests/test_nlp.py::test_load_sentiment_dict_success`は`assets/sentiment_dict.csv`の存在を前提とするが、CI(`.github/workflows/tests.yml`)は`python scripts/download_sentiment.py \|\| true`で取得失敗を握り潰すため、**辞書取得に失敗したrunではテストが落ちる**。`skipif`で辞書の有無に応じてスキップさせるか、`\|\| true`を外して取得失敗そのものを検知させるかを決める必要がある |
| B | `networkx`/`openpyxl`が未ピン留め | 判断待ち（要bat実行） | `requirements.txt`末尾の`networkx>=3.0`/`openpyxl>=3.0`だけが`==`ではなく`>=`（他は全て`==`）。手書き追記の痕跡で、実際に`.venv`へ未インストールだった原因でもある（発見事項1）。`scripts/create_requirements.bat`を実行すれば`.venv`の実態から再生成されて解消するが、ローカル環境でbatを走らせる操作のため保留 |

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
- ✅ **対応済み** ~~感情辞書ダウンロード失敗時の挙動を明記~~: 辞書なしでもアプリは起動し、`load_sentiment_dict()`が空dictを返す→感情スコア`0.0`・分類「ニュートラル」にフォールバックすることをCLAUDE.mdとREADMEに記載した（2026-09-27）。
- ✅ **対応済み** ~~requirements.txtのビルド専用パッケージ分離~~: `requirements-build.txt`（`-r requirements.txt` + `pyinstaller`/`pyinstaller-hooks-contrib`/`pefile`/`pywin32-ctypes`/`altgraph`）を新設し、`requirements.txt`からは除去した。`scripts/build.bat`はビルド前に`requirements-build.txt`をインストールし、`scripts/create_requirements.bat`は`findstr`でビルド専用パッケージを除外して再生成する（2026-09-27）。
- ✅ **対応済み** ~~CLAUDE.mdにsrc/core/配下のモジュール構成を追記~~: `## モジュール構成（src/）`を追加し、`config.py`/`nlp_engine.py`/`stats.py`/`visualizer.py`（と`run.py`）の役割を表で明記した（2026-09-27）。

### 低
- ✅ **対応済み** ~~感情辞書のダウンロードURLをHTTPS化できないか確認する~~: 提供元サイトはHTTPSに対応していた（名詞編・用言編とも200、取得バイト数はHTTPと同一）。`scripts/download_sentiment.py`のURLを`https://`に変更し、実際に18,528語の辞書を取得できることを確認した（2026-09-27）。
- ✅ **対応済み** ~~`TextMiningApp.spec`/`scripts/build.bat`によるexeビルドは「ローカル/CI専用作業であり、クラウドセッションでは通常触らない」旨をCLAUDE.mdに一言添える~~: CLAUDE.mdの注意点に、ビルドはローカル/CI専用・`pip install -r requirements-build.txt`が必要・成果物は`dist/`と`build/`（gitignore対象）である旨を追記した（2026-09-27）。

## 対応時に発見した追加事項（2026-09-27）

監査項目の対応中に見つかった、上記7件には含まれない問題。

1. **`.venv`に`networkx`と`openpyxl`が未インストールだった** — 両者は`requirements.txt`に記載があるのに実際には入っておらず、`src/core/visualizer.py:5`が`import networkx`をモジュールレベルで行うため、**アプリが起動時に`ModuleNotFoundError`で落ちる状態だった**。`pip install -r requirements.txt`で復旧済み（他は全て充足していたため追加インストールは6パッケージのみ）。
   - 原因の推測: `requirements.txt`末尾の`networkx>=3.0`と`openpyxl>=3.0`だけが`==`ではなく`>=`で、`pip freeze`由来ではなく**手書きで追記された痕跡**がある。追記時にインストールが漏れた可能性が高い。
   - 再発防止: 依存を追加する際は`pip install`と`requirements.txt`の更新をセットで行う。`scripts/create_requirements.bat`を使えば`.venv`の実態から再生成されるため、この種のズレは起きない。→ **残課題Bとして起票**（`requirements.txt`の`>=`指定が残っている）。
2. **`tests/test_nlp.py::test_load_sentiment_dict_success`は辞書ファイルに依存する** — `assets/sentiment_dict.csv`はgitignore対象のため、未ダウンロードの環境では必ず失敗する。実際、対応前は`1 failed, 41 passed`だった。辞書をダウンロード後は`42 passed`で全面グリーン。
   - CI(`.github/workflows/tests.yml`)は`python scripts/download_sentiment.py || true`を実行してからテストするため、**辞書の取得に失敗した場合はこのテストが落ちる**（`|| true`でダウンロード失敗を握り潰しているのに、テスト側は辞書の存在を前提にしている）。CIを安定させるなら、このテストを`skipif`で辞書の有無に応じてスキップさせるか、`|| true`を外して失敗を検知させるかの判断が必要。→ **残課題Aとして起票**。
3. **READMEのテストまわりの記載が古かった**（今回修正） — `tests/`配下には実際には`test_config.py`/`test_nlp.py`/`test_stats.py`/`test_visualizer.py`の4ファイル（計42テスト）があるが、READMEのディレクトリ構成は`test_nlp.py`と`test_stats.py`の2つしか挙げていなかった。また実行コマンドも`pytest`とだけ書かれていたが、これは`No module named 'src'`で失敗する（実測確認済み）。`PYTHONPATH=. pytest tests/ -v`に修正した。

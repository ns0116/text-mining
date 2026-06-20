# テキストマイニング・アプリケーション

日本語テキストの形態素解析を行い、頻出単語や重要キーワード（TF-IDF）を抽出して、棒グラフやワードクラウドで可視化する Streamlit アプリケーションです。

本プロジェクトはプロフェッショナル版 (v2.0) として高度化されており、属性別セグメント分析、共起ネットワーク、感情分析、N-gram、対応分析（Correspondence Analysis）など、自然言語学者やマーケター向けの高度な分析が可能です。

---

## 📁 ディレクトリ構成

```text
text-mining/
│  .gitignore                   # Git除外設定
│  README.md                    # 本説明書
│  requirements.txt             # 依存ライブラリ一覧
│  TextMiningApp.spec           # PyInstallerビルド構成ファイル
│  text_mining_config.json      # アプリ設定ファイル (Git除外)
│
├─assets/                       # リソースフォルダ
│      NotoSansJP-Regular.ttf   # 日本語フォントファイル
│      sample.txt               # 動作確認用サンプルテキスト
│      sentiment_dict.csv       # 感情極性辞書（自動ダウンロード）
│
├─src/                          # ソースコードフォルダ
│  │  text_mining_app.py       # メインアプリケーションコード (Streamlit UI)
│  │  run.py                   # exe起動用エントリーポイント
│  │
│  └─core/                     # コアロジックモジュール
│          __init__.py          # パッケージ初期化
│          config.py            # 設定値・セッション管理
│          nlp_engine.py        # 自然言語処理・形態素解析エンジン
│          stats.py             # 統計・データ抽出（TF-IDF, N-gram, 共起, 対応分析等）
│          visualizer.py        # 可視化処理（WordCloud, 共起ネットワーク等の描画）
│
├─scripts/                      # 各種スクリプトフォルダ
│      text_mining_app_launch.bat # アプリ通常起動バッチ (Windows用)
│      text_mining_app_launch.sh  # アプリ起動スクリプト (macOS用)
│      build.bat                # exeビルド実行用バッチ (Windows用)
│      create_requirements.bat  # requirements.txt自動更新バッチ
│      download_sentiment.py    # 感情極性辞書ダウンロードスクリプト
│
├─tests/                        # テストコードフォルダ
│      test_nlp.py              # 自然言語処理エンジンのテスト
│      test_stats.py            # 統計分析処理のテスト
│
├─.venv/                        # Python仮想環境 (Git除外)
└─dist/                         # ビルドされた成果物出力先 (Git除外)
```

---

## 🛠️ 開発環境での動かし方

### 1. 依存ライブラリのインストール
ターミナル（仮想環境を有効にした状態）でプロジェクトのルートフォルダから以下を実行し、必要なパッケージをインストールします。

```bash
pip install -r requirements.txt
```

### 2. 自然言語処理モデル (GiNZA) のダウンロード
日本語の形態素解析モデル（GiNZA）をインストールします。

```bash
python -m spacy download ja_ginza
```

### 3. 感情極性辞書のダウンロード
感情分析機能を利用するため、事前に感情極性辞書（東北大学 乾・関根研究室）をダウンロードして `assets/` 配下に配置します。
以下のスクリプトを実行することで、自動的にダウンロードと加工・配置が行われます。

```bash
python scripts/download_sentiment.py
```

### 4. アプリの起動

#### macOS の場合:
ターミナルで `scripts/` フォルダ内の以下のシェルスクリプトを実行します。自動的に仮想環境の検出、依存関係のチェック、およびアプリの起動が行われます。

```bash
chmod +x scripts/text_mining_app_launch.sh
./scripts/text_mining_app_launch.sh
```

#### Windows の場合:
`scripts/` フォルダ内の以下のバッチファイルをダブルクリックして起動します。

* **`scripts/text_mining_app_launch.bat`**

起動後、自動的にブラウザが開いて `http://localhost:8501` でアプリが利用可能になります。

---

## 🧪 テストの実行方法

コアロジックの動作検証のために `pytest` によるユニットテストを実行できます。

### 1. pytest のインストール
```bash
pip install pytest
```

### 2. テストの実行
プロジェクトルートディレクトリで以下を実行します。

```bash
pytest
```

---

## 📦 実行ファイル (exe / バイナリ) のビルド方法

本アプリは、Python環境が入っていないPCでも動作するスタンドアロンな実行ファイルにビルドできます。

### Windows の場合:
`scripts/` フォルダ内の以下のバッチファイルをダブルクリックして実行します。
* **`scripts/build.bat`**

### macOS の場合:
ターミナル（仮想環境を有効にした状態）で、プロジェクトルートフォルダから以下を実行します。

```bash
pyinstaller TextMiningApp.spec --clean
```

### 成果物の確認
ビルド完了後、**`dist/`** フォルダの中に実行ファイル（Windowsは `TextMiningApp.exe`）が生成されます。このファイルをダブルクリックするだけで、Python環境がないPCでも自動的にブラウザが立ち上がり、アプリを使用できます。


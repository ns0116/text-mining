# テキストマイニング・アプリケーション

日本語テキストの形態素解析を行い、頻出単語や重要キーワード（TF-IDF）を抽出して、棒グラフやワードクラウドで可視化する Streamlit アプリケーションです。

本プロジェクトは、ソースコードが **`src/`** 、バッチファイルが **`scripts/`** 、データやフォントなどのリソースが **`assets/`** に整理されて設計されています。

---

## 📁 ディレクトリ構成

```text
Text_Mining/
│  .gitignore                   # Git除外設定
│  README.md                    # 本説明書
│  requirements.txt             # 依存ライブラリ一覧
│  TextMiningApp.spec           # PyInstallerビルド構成ファイル
│
├─assets/                       # リソースフォルダ
│      NotoSansJP-Regular.ttf   # 日本語フォントファイル
│      sample.txt               # 動作確認用サンプルテキスト
│
├─src/                          # ソースコードフォルダ
│      text_mining_app.py       # メインアプリケーションコード
│      run.py                   # exe起動用エントリーポイント
│
├─scripts/                      # 各種バッチファイルフォルダ
│      text_mining_app_launch.bat # アプリ通常起動バッチ
│      build.bat                # exeビルド実行用バッチ
│      create_requirements.bat  # requirements.txt自動更新バッチ
│
├─.venv/                        # Python仮想環境 (Git除外)
├─_attic/                       # 過去の古いバックアップ退避フォルダ
└─dist/                         # ビルドされたexe出力先 (Git除外)
```

---

## 🛠️ 開発環境での動かし方

### 1. 依存ライブラリのインストール
ターミナル（仮想環境を有効にした状態）でプロジェクトのルートフォルダから以下を実行し、必要なパッケージをインストールします。

```bash
pip install -r requirements.txt
```

### 2. GiNZAモデルのダウンロード
日本語の形態素解析モデル（GiNZA）をインストールします。

```bash
python -m spacy download ja_ginza
```

### 3. アプリの起動
`scripts/` フォルダ内の以下のバッチファイルをダブルクリックして起動します。
* **`scripts/text_mining_app_launch.bat`**

※ バッチファイルは内部で自動的に親ディレクトリ（プロジェクトルート）に移動して実行します。

起動後、自動的にブラウザが開いて `http://localhost:8501` でアプリが利用可能になります。

---

## 📦 実行ファイル (exe) のビルド方法

本アプリは、Python環境が入っていない他のPCでも動作するスタンドアロンな `TextMiningApp.exe` にビルドできます。

### 1. ビルドの実行
`scripts/` フォルダ内の以下のバッチファイルをダブルクリックして実行します。
* **`scripts/build.bat`**

※ ビルドには spacy やモデルデータのコピーが含まれるため、数分かかる場合があります。

### 2. 成果物の確認
ビルド完了後、**`dist/`** フォルダの中に **`TextMiningApp.exe`** が生成されます。
このファイルをダブルクリックするだけで、Python環境がない環境でも自動的にブラウザが立ち上がり、アプリを使用できます。

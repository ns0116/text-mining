# ファイル名: run.py
# 役割: PyInstallerで作成した実行可能ファイルのエントリーポイント。
#       Streamlitアプリケーションを正しい方法で起動する。

import sys
import os
import webbrowser
import multiprocessing
from streamlit.web import cli as stcli

if __name__ == '__main__':
    # 実行可能ファイル（--onefile）として実行された際に、
    # 新しいプロセスを生成することによる無限ループを防ぐための標準的なおまじない。
    multiprocessing.freeze_support()

    # PyInstallerによって一時的に展開された場所（_MEIPASS）を取得する。
    # これにより、同梱されたファイル（app.pyなど）へのパスを正しく解決できる。
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    # プロセスのカレントディレクトリを、ファイルが展開された場所に変更する。
    # これにより、streamlitが 'app.py' を相対パスで見つけられるようになる。
    os.chdir(base_path)

    # 実行するメインのアプリファイル名
    app_file = "text_mining_app.py"

    # Streamlitをコマンドラインからではなく、スクリプトから起動するための準備。
    # `streamlit run app.py` と同等の引数をプログラム的に作成する。
    sys.argv = [
        "streamlit", "run", app_file,
        "--global.developmentMode", "false",  # 実行時エラーを防ぐための設定
        "--server.headless", "true",        # サーバー自身がブラウザを開くのを防ぐ
        "--server.port", "8501",            # 使用するポート番号を固定
        "--server.fileWatcherType", "none"  # ファイルの変更監視を無効化
    ]
    
    # ユーザーのために、デフォルトのブラウザでアプリケーションのURLを開く
    webbrowser.open("http://localhost:8501")
    
    # Streamlitのメインのコマンドラインインターフェース関数を実行し、サーバーを起動する
    stcli.main()
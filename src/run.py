# ファイル名: run.py
# 役割: PyInstallerで作成した実行可能ファイルのエントリーポイント。
#       Streamlitアプリケーションを正しい方法で起動する。

import sys
import os
import webbrowser
import multiprocessing
import socket
import threading
import time
from streamlit.web import cli as stcli

def is_port_in_use(port):
    """ポートが使用中であるか確認する"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port=8501):
    """8501から開始し、使用可能な最初のポートを探す"""
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port

def wait_and_open_browser(port):
    """サーバーが起動してポートが有効になるまで待機し、ブラウザを開く"""
    for _ in range(100):  # 最大10秒間待機
        if is_port_in_use(port):
            webbrowser.open(f"http://localhost:{port}")
            return
        time.sleep(0.1)

if __name__ == '__main__':
    # 実行可能ファイル（--onefile）として実行された際に、
    # 新しいプロセスを生成することによる無限ループを防ぐための標準的なおまじない。
    multiprocessing.freeze_support()

    # PyInstallerによって一時的に展開された場所（_MEIPASS）を取得する。
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    # プロセスのカレントディレクトリを、ファイルが展開された場所に変更する。
    os.chdir(base_path)

    # 実行するメインのアプリファイル名
    app_file = "text_mining_app.py"

    # 使用可能なポートの動的取得
    port = find_available_port(8501)

    # Streamlitをスクリプトから起動するための引数作成
    sys.argv = [
        "streamlit", "run", app_file,
        "--global.developmentMode", "false",  # 実行時エラーを防ぐ設定
        "--server.headless", "true",        # サーバー自身がブラウザを開くのを防ぐ
        "--server.port", str(port),         # 動的に割り当てられたポートを指定
        "--server.fileWatcherType", "none"  # ファイルの変更監視を無効化
    ]
    
    # サーバーの立ち上がりを待機してブラウザを開くスレッドを起動
    browser_thread = threading.Thread(target=wait_and_open_browser, args=(port,))
    browser_thread.daemon = True
    browser_thread.start()
    
    # Streamlitのメインのコマンドラインインターフェース関数を実行し、サーバーを起動する
    stcli.main()
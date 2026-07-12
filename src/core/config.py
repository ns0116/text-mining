import os
import sys
import json
import pandas as pd
import streamlit as st

def resource_path(relative_path):
    """ 実行ファイル内のリソースへのパスを取得する """
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_application_path():
    """ 実行ファイルの場所、またはスクリプトの場所を取得する """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return os.path.dirname(sys.executable)
    core_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(core_dir))

# --- 定数 ---
CONFIG_FILE = os.path.join(get_application_path(), "text_mining_config.json")
DEFAULT_STOP_WORDS = ['こと', 'もの', 'よう', 'ため', 'これ', 'それ', 'あれ', 'さん', 'する', 'いる', 'なる', 'ある', 'ない', 'いう', '思う', 'できる', 'とき', 'ところ']
MAX_CHARS_PER_CHUNK = 10000

# 詳細な品詞定義
POS_MAP_JAPANESE = {
    '名詞 (一般)': 'NOUN',
    '固有名詞': 'PROPN',
    '代名詞': 'PRON',
    '動詞': 'VERB',
    '形容詞': 'ADJ',
    '副詞': 'ADV',
    '接続詞': 'CONJ',
    '助動詞': 'AUX',
    '助詞': 'ADP'
}
POS_MAP_JAPANESE_REV = {v: k for k, v in POS_MAP_JAPANESE.items()}

def get_system_font_options():
    """システム上の日本語フォント候補を検索し、有効なパスと名称のリストを返す"""
    standard_fonts = {
        "Noto Sans JP (添付)": resource_path("assets/NotoSansJP-Regular.ttf"),
        "ヒラギノ角ゴ (Mac)": "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "ヒラギノ明朝 (Mac)": "/System/Library/Fonts/ヒラギノ明朝 ProN.ttc",
        "游ゴシック (Windows)": "C:/Windows/Fonts/YuGothM.ttc",
        "メイリオ (Windows)": "C:/Windows/Fonts/meiryo.ttc",
        "ＭＳ ゴシック (Windows)": "C:/Windows/Fonts/msgothic.ttc",
        "Noto Sans CJK (Linux)": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "IPAゴシック (Linux)": "/usr/share/fonts/truetype/fonts-ipafont-gothic/ipag.ttf"
    }
    valid_fonts = {}
    for name, path in standard_fonts.items():
        if os.path.exists(path):
            valid_fonts[name] = path
    return valid_fonts

def load_config():
    """設定ファイルを読み込む"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # Removed st.warning to avoid UI dependency
            print(f"設定ファイルの読み込みに失敗しました: {e}")
    return {}

def save_config(config_data):
    """設定ファイルを保存する"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        # Removed st.error to avoid UI dependency
        print(f"設定ファイルの保存に失敗しました: {e}")

def initialize_session_state():
    """セッションステートを初期化する"""
    if 'initialized' in st.session_state:
        return

    config = load_config()
    st.session_state.stop_words = config.get("stop_words", DEFAULT_STOP_WORDS)
    st.session_state.top_n = config.get("top_n", 10)
    st.session_state.font_path = config.get("font_path", "")
    st.session_state.selected_pos = config.get("selected_pos", ['NOUN', 'PROPN', 'VERB', 'ADJ'])
    st.session_state.synonyms_text = config.get("synonyms_text", "")
    st.session_state.document_resolution = config.get("document_resolution", "行単位（文単位）")
    st.session_state.sentiment_threshold = config.get("sentiment_threshold", 0.05)
    st.session_state.ngram_min_count = config.get("ngram_min_count", 2)
    
    st.session_state.raw_tokens = None
    st.session_state.raw_sentences = None
    
    st.session_state.input_text = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"
    st.session_state.import_type = "直接入力 / テキストファイル"
    st.session_state.df_uploaded = None
    st.session_state.text_col = ""
    st.session_state.attr_col = None
    
    st.session_state.analysis_complete = False
    st.session_state.df_freq = pd.DataFrame()
    st.session_state.df_tfidf = pd.DataFrame()
    st.session_state.df_ngrams = pd.DataFrame()
    st.session_state.df_edges = pd.DataFrame()
    st.session_state.df_sentences = pd.DataFrame()
    st.session_state.df_tokens = pd.DataFrame()
    st.session_state.df_ca = None
    st.session_state.corpus_stats = {}

    # 初回起動時に設定ファイルがなければ作成
    if not os.path.exists(CONFIG_FILE):
        save_config({
            "top_n": st.session_state.top_n,
            "font_path": st.session_state.font_path,
            "stop_words": st.session_state.stop_words,
            "selected_pos": st.session_state.selected_pos,
            "synonyms_text": st.session_state.synonyms_text,
            "document_resolution": st.session_state.document_resolution,
            "sentiment_threshold": st.session_state.sentiment_threshold,
            "ngram_min_count": st.session_state.ngram_min_count
        })
    
    st.session_state.initialized = True

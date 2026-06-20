# ファイル名: text_mining_app.py
# 説明: app.pyの機能改善を取り込み、PyInstaller依存をなくしたバージョン。可読性、堅牢性、UIを改善。
# Version: v15 (レビュー・修正版)
# 
# --- 実行前の準備 (ターミナルで実行) ---
# 1. 必要なライブラリをインストール:
# pip install -r requirements.txt
#
# 2. GiNZAのモデルをインストール:
# python -m spacy download ja_ginza
#
# --- requirements.txt の内容 ---
# streamlit
# spacy
# ja-ginza
# pandas
# plotly
# wordcloud
# matplotlib
# scikit-learn
#
# --- 実行方法 ---
# streamlit run text_mining_app.py

import streamlit as st
import spacy
from collections import Counter
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import os
import datetime
import io
import time
import sys

# --------------------------------------------------------------------------
# 1. PyInstaller対応ヘルパー関数 & 初期設定
# --------------------------------------------------------------------------

def resource_path(relative_path):
    """ 実行ファイル内のリソースへのパスを取得する """
    if hasattr(sys, '_MEIPASS'):
        # PyInstallerによって作成された一時フォルダ内のパスを返す
        base_path = sys._MEIPASS
    else:
        # 通常のPython環境では、このファイルの場所を基準とする
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_application_path():
    """ 実行ファイルの場所、またはスクリプトの場所を取得する """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # 実行ファイルとして実行されている場合
        return os.path.dirname(sys.executable)
    # スクリプトとして実行されている場合（src/ の親フォルダ、すなわちプロジェクトルートを返す）
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 定数 ---
CONFIG_FILE = os.path.join(get_application_path(), "text_mining_config.json")
DEFAULT_STOP_WORDS = ['こと', 'もの', 'よう', 'ため', 'これ', 'それ', 'あれ', 'さん', 'する', 'いる', 'なる', 'ある', 'ない', 'いう', '思う', 'できる', 'とき', 'ところ']
MAX_CHARS_PER_CHUNK = 10000  # 大きなテキストに対応するためのチャンクサイズ
TARGET_POS = ['NOUN', 'VERB', 'ADJ'] # 解析対象とする品詞 (名詞, 動詞, 形容詞)
POS_MAP_JAPANESE = {'名詞': 'NOUN', '動詞': 'VERB', '形容詞': 'ADJ'}

# --- Streamlit ページ設定 ---
st.set_page_config(layout="wide", page_title="テキストマイニング・アプリ")


# --------------------------------------------------------------------------
# 2. ユーティリティ関数 & 設定管理
# --------------------------------------------------------------------------

def load_config():
    """設定ファイルを読み込む"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            st.warning(f"設定ファイルの読み込みに失敗しました: {e}")
    return {}

def save_config(config_data):
    """設定ファイルを保存する"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        st.error(f"設定ファイルの保存に失敗しました: {e}")

def initialize_session_state():
    """セッションステートを初期化する"""
    if 'initialized' in st.session_state:
        return

    config = load_config()
    st.session_state.stop_words = config.get("stop_words", DEFAULT_STOP_WORDS)
    st.session_state.top_n = config.get("top_n", 10)
    st.session_state.font_path = config.get("font_path", "")
    st.session_state.input_text = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"
    
    st.session_state.analysis_complete = False
    st.session_state.df_freq = pd.DataFrame()
    st.session_state.df_tfidf = pd.DataFrame()

    # 初回起動時に設定ファイルがなければ作成
    if not os.path.exists(CONFIG_FILE):
        save_config({
            "top_n": st.session_state.top_n,
            "font_path": st.session_state.font_path,
            "stop_words": st.session_state.stop_words
        })
    
    st.session_state.initialized = True

# --------------------------------------------------------------------------
# 3. 解析エンジン・コア機能
# --------------------------------------------------------------------------

@st.cache_resource
def load_ginza_model():
    """GiNZAモデルをキャッシュして読み込む。実行ファイル化に対応。"""
    try:
        # 1. 同梱されたモデルのパスからロードを試みる (PyInstallerビルド用)
        model_path = resource_path('ja_ginza')
        return spacy.load(model_path)
    except OSError:
        # 2. 通常のPython環境からのロードを試みる
        try:
            return spacy.load('ja_ginza')
        except OSError:
            st.error("GiNZAモデルが見つかりません。")
            st.error("開発環境では `python -m spacy download ja_ginza` を実行してください。")
            st.error("実行ファイルを作成する際は、`--add-data` オプション等でモデルを同梱する必要があります。")
            st.stop()

nlp = load_ginza_model()

def perform_full_analysis(text):
    """テキストの形態素解析、頻出度分析、TF-IDF分析をまとめて実行し、進捗を表示する"""
    progress_bar = st.progress(0, text="テキストの前処理中...")
    text_to_analyze = text.replace('\n', '').replace('\r', '')

    all_tokens = []
    text_chunks = [text_to_analyze[i:i + MAX_CHARS_PER_CHUNK] for i in range(0, len(text_to_analyze), MAX_CHARS_PER_CHUNK)]
    total_chunks = len(text_chunks) if len(text_chunks) > 0 else 1

    for i, chunk in enumerate(text_chunks):
        doc = nlp(chunk)
        for token in doc:
            if token.pos_ in TARGET_POS and token.lemma_ not in st.session_state.stop_words and len(token.lemma_) > 0:
                all_tokens.append({'単語': token.lemma_, '品詞': token.pos_})
        progress_percentage = 10 + int(60 * (i + 1) / total_chunks)
        progress_bar.progress(progress_percentage, text=f"形態素解析中... ({i + 1}/{total_chunks}チャンク完了)")

    if not all_tokens:
        progress_bar.empty()
        st.warning("解析可能な単語が見つかりませんでした。テキストの内容やストップワードの設定を確認してください。")
        return None, None

    df_all_words = pd.DataFrame(all_tokens)

    progress_bar.progress(75, text="頻出度を集計中...")
    df_freq = df_all_words.groupby(['単語', '品詞']).size().reset_index(name='出現回数').sort_values(by='出現回数', ascending=False)

    progress_bar.progress(85, text="重要度（TF-IDF）を計算中...")
    def tokenize_for_tfidf(t):
        doc = nlp(t)
        return [token.lemma_ for token in doc if token.pos_ in TARGET_POS and token.lemma_ not in st.session_state.stop_words]

    sentences = [s.strip() for s in text_to_analyze.strip().split('。') if s.strip()]
    df_tfidf = pd.DataFrame()
    if len(sentences) > 1:
        try:
            vectorizer = TfidfVectorizer(tokenizer=tokenize_for_tfidf, token_pattern=None)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            avg_tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            df_tfidf = pd.DataFrame({'単語': feature_names, '平均重要度スコア': avg_tfidf_scores})
            
            # 品詞情報をマージ
            pos_info = df_all_words[['単語', '品詞']].drop_duplicates(subset='単語')
            df_tfidf = pd.merge(df_tfidf, pos_info, on='単語', how='left').sort_values(by='平均重要度スコア', ascending=False)
        except Exception as e:
            st.warning(f"TF-IDFの計算中にエラーが発生しました。文の数が少ないか、特異なデータの場合に発生することがあります。 (詳細: {e})")

    progress_bar.progress(100, text="解析完了！")
    time.sleep(0.5)
    progress_bar.empty()

    return df_freq, df_tfidf

def generate_wordcloud_fig(frequencies, font_path):
    """ワードクラウドの画像を生成する。フォントパスを自動検索。"""
    if not frequencies:
        return None
    
    font_path_to_use = None
    # 1. ユーザー指定のパスを最優先
    if font_path and os.path.exists(font_path):
        font_path_to_use = font_path
    else:
        # 2. 同梱フォントを探す (PyInstaller想定)
        try:
            bundled_font = resource_path('assets/NotoSansJP-Regular.ttf')
            if os.path.exists(bundled_font):
                font_path_to_use = bundled_font
        except Exception:
            pass

    # 3. 同梱フォントが見つからない場合、システムフォントを探す
    if not font_path_to_use or not os.path.exists(font_path_to_use):
        font_paths = [
            'C:/Windows/Fonts/YuGothM.ttc',
            'C:/Windows/Fonts/meiryo.ttc',
            '/System/Library/Fonts/Hiragino Sans GB.ttc',
            '/System/Library/Fonts/AppleGothic.ttf',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        ]
        for path in font_paths:
            if os.path.exists(path):
                font_path_to_use = path
                break
        else:
            st.warning("適切な日本語フォントが見つかりません。ワードクラウドが文字化けする可能性があります。")
            font_path_to_use = None

    try:
        wc = WordCloud(font_path=font_path_to_use, 
                       width=800, height=600, background_color='white', 
                       max_words=150, collocations=False, max_font_size=100
                       ).generate_from_frequencies(frequencies)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception as e:
        st.error(f"ワードクラウドの描画中にエラーが発生しました: {e}")
        return None

# --------------------------------------------------------------------------
# 4. UI操作用コールバック関数
# --------------------------------------------------------------------------

def set_analysis_flag_off():
    """解析状態をリセットする"""
    st.session_state.analysis_complete = False

def update_text_from_upload():
    """アップロードされたファイルからテキストを読み込む。複数のエンコーディングを試す。"""
    uploaded_file = st.session_state.get('file_uploader_widget')
    if uploaded_file:
        encodings = ['utf-8', 'shift-jis', 'cp932']
        decoded = False
        for enc in encodings:
            try:
                st.session_state.input_text = uploaded_file.read().decode(enc)
                uploaded_file.seek(0) # 次回の読み込みのためにポインタを戻す
                decoded = True
                break
            except UnicodeDecodeError:
                continue
        
        if decoded:
            set_analysis_flag_off()
        else:
            st.error("ファイルのデコードに失敗しました。対応エンコーディング: utf-8, shift-jis, cp932")

def handle_text_area_change():
    """テキストエリアの変更をsession_stateに反映する"""
    st.session_state.input_text = st.session_state.get('text_area_widget', '')
    set_analysis_flag_off()

def clear_input_text():
    """入力テキストをクリアする"""
    st.session_state.input_text = ""
    set_analysis_flag_off()

# --------------------------------------------------------------------------
# 5. Streamlit UI構築 (メイン処理)
# --------------------------------------------------------------------------

def main():
    """アプリケーションのメインUIとロジックを構築する"""
    initialize_session_state()

    st.title("📝 テキストマイニング・アプリケーション")

    # --- サイドバー ---
    with st.sidebar:
        st.header("⚙️ 設定")
        st.info("設定は「解析実行」時に自動で保存されます。")
        
        st.session_state.top_n = st.slider(
            "グラフに表示する単語の数", min_value=5, max_value=50, 
            value=st.session_state.top_n, on_change=set_analysis_flag_off
        )
        st.session_state.font_path = st.text_input(
            "ワードクラウド用フォントパス", value=st.session_state.font_path,
            placeholder='例: C:/Windows/Fonts/YuGothM.ttc',
            on_change=set_analysis_flag_off,
            help="未入力の場合、システムフォントを自動検索します。"
        )
        
        st.header("🚫 ストップワード設定")
        st.session_state.stop_words = [
            word.strip() for word in st.text_area(
                "除外したい単語（1行に1つ）", value="\n".join(st.session_state.stop_words),
                height=250, help="ここに追加した単語は、すべての解析から除外されます。",
                on_change=set_analysis_flag_off
            ).split('\n') if word.strip()
        ]

    # --- メインコンテンツ ---
    st.header("1. テキストの入力")
    st.file_uploader(
        "テキストファイル (.txt) をアップロード", type=['txt'], 
        key='file_uploader_widget', on_change=update_text_from_upload
    )
    st.text_area(
        "または、ここにテキストを直接入力", value=st.session_state.input_text, 
        key='text_area_widget', on_change=handle_text_area_change, height=250
    )

    if st.button("入力内容をクリア", on_click=clear_input_text):
        st.rerun()

    st.header("2. 解析の実行")
    if st.button("解析実行", type="primary"):
        if st.session_state.input_text.strip():
            df_freq, df_tfidf = perform_full_analysis(st.session_state.input_text)
            if df_freq is not None:
                st.session_state.df_freq = df_freq
                st.session_state.df_tfidf = df_tfidf if df_tfidf is not None else pd.DataFrame()
                st.session_state.analysis_complete = True
                
                # 設定を保存
                save_config({
                    "top_n": st.session_state.top_n,
                    "font_path": st.session_state.font_path,
                    "stop_words": st.session_state.stop_words
                })
                st.toast("設定を保存しました。")
                st.rerun()
            else:
                st.session_state.analysis_complete = False
        else:
            st.error("テキストが入力されていません。")
            st.session_state.analysis_complete = False

    # --- 解析結果表示エリア ---
    if st.session_state.analysis_complete:
        st.success("解析が完了しました。下のタブで結果を確認できます。")
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        tab1, tab2 = st.tabs(["📊 頻出度分析", "📈 重要度分析 (TF-IDF)"])

        # --- 頻出度分析タブ ---
        with tab1:
            st.info("テキスト内での単語の**出現回数**に基づいた分析です。")
            
            selected_pos_jp = st.multiselect(
                "品詞で絞り込み:", options=list(POS_MAP_JAPANESE.keys()), 
                default=list(POS_MAP_JAPANESE.keys()), key='freq_pos_filter'
            )
            selected_pos_codes = [POS_MAP_JAPANESE[p] for p in selected_pos_jp]
            
            st.markdown("---")
            df_freq_all = st.session_state.df_freq
            df_display = df_freq_all[df_freq_all['品詞'].isin(selected_pos_codes)]
            
            if df_display.empty:
                st.warning(f"選択された品詞の単語が見つかりませんでした。")
            else:
                df_display = df_display.reset_index(drop=True)
                word_counts = dict(zip(df_display['単語'], df_display['出現回数']))
                
                col1, col2 = st.columns([0.55, 0.45])
                with col1:
                    st.subheader(f"棒グラフ")
                    fig_bar = px.bar(df_display.head(st.session_state.top_n), x='単語', y='出現回数', text_auto=True).update_layout(xaxis_title=None, yaxis_title="出現回数", xaxis={'categoryorder':'total descending'})
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.download_button(label="📥 グラフをPNGで保存", data=fig_bar.to_image(format="png"), file_name=f"freq_bar_{timestamp}.png", mime="image/png")
                
                with col2:
                    st.subheader(f"ワードクラウド")
                    fig_wc = generate_wordcloud_fig(word_counts, st.session_state.font_path)
                    if fig_wc:
                        st.pyplot(fig_wc)
                        buf = io.BytesIO(); fig_wc.savefig(buf, format="png", bbox_inches='tight')
                        st.download_button(label="📥 ワードクラウドをPNGで保存", data=buf.getvalue(), file_name=f"freq_wc_{timestamp}.png", mime="image/png")
                
                st.subheader("解析データ")
                st.download_button(label="📥 全データをCSVでダウンロード", data=df_display.to_csv(index=False, encoding='utf-8-sig'), file_name=f"freq_data_{timestamp}.csv", mime='text/csv')
                st.dataframe(df_display.rename(columns={'品詞': '品詞コード'}), height=300, use_container_width=True)

        # --- 重要度分析タブ ---
        with tab2:
            st.info("テキストの文脈を考慮し、内容の核となる**重要キーワード**をスコア化して分析します。")
            
            selected_pos_jp_tfidf = st.multiselect(
                "品詞で絞り込み:", options=list(POS_MAP_JAPANESE.keys()), 
                default=list(POS_MAP_JAPANESE.keys()), key='tfidf_pos_filter'
            )
            selected_pos_codes_tfidf = [POS_MAP_JAPANESE[p] for p in selected_pos_jp_tfidf]
            
            st.markdown("---")
            df_tfidf_all = st.session_state.df_tfidf
            
            if df_tfidf_all.empty:
                st.warning("重要度（TF-IDF）の計算結果がありません。")
            else:
                df_filtered = df_tfidf_all[df_tfidf_all['品詞'].isin(selected_pos_codes_tfidf)]
                if df_filtered.empty:
                    st.warning(f"選択された品詞の重要単語が見つかりませんでした。")
                else:
                    df_filtered = df_filtered.reset_index(drop=True)
                    col1, col2 = st.columns([0.55, 0.45])
                    with col1:
                        st.subheader(f"棒グラフ")
                        fig_bar_tfidf = px.bar(df_filtered.head(st.session_state.top_n), x='単語', y='平均重要度スコア', text_auto='.3f').update_layout(xaxis_title=None, yaxis_title="平均重要度スコア", xaxis={'categoryorder':'total descending'})
                        st.plotly_chart(fig_bar_tfidf, use_container_width=True)
                        st.download_button(label="📥 グラフをPNGで保存", data=fig_bar_tfidf.to_image(format="png"), file_name=f"tfidf_bar_{timestamp}.png", mime="image/png")
                    
                    with col2:
                        st.subheader(f"ワードクラウド")
                        # スコアを強調するためにスケールアップ
                        frequencies_dict = {row['単語']: row['平均重要度スコア'] * 1000 for index, row in df_filtered.iterrows() if row['平均重要度スコア'] > 0}
                        fig_wc_tfidf = generate_wordcloud_fig(frequencies_dict, st.session_state.font_path)
                        if fig_wc_tfidf:
                            st.pyplot(fig_wc_tfidf)
                            buf_tfidf = io.BytesIO(); fig_wc_tfidf.savefig(buf_tfidf, format="png", bbox_inches='tight')
                            st.download_button(label="📥 ワードクラウドをPNGで保存", data=buf_tfidf.getvalue(), file_name=f"tfidf_wc_{timestamp}.png", mime="image/png")

                    st.subheader("解析データ")
                    df_display_tfidf = df_filtered.rename(columns={'品詞': '品詞コード'})
                    st.download_button(label="📥 全データをCSVでダウンロード", data=df_display_tfidf.to_csv(index=False, encoding='utf-8-sig'), file_name=f"tfidf_data_{timestamp}.csv", mime='text/csv')
                    st.dataframe(df_display_tfidf[['単語', '平均重要度スコア', '品詞コード']], height=300, use_container_width=True)

    elif 'initialized' in st.session_state and not st.session_state.analysis_complete:
        st.info("テキストを入力・編集し、「解析実行」ボタンを押してください。")


if __name__ == "__main__":
    main()

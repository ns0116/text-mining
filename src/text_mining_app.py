# ファイル名: text_mining_app.py
# 説明: 自然言語学者およびプロのマーケター向けに高度化したテキストマイニングアプリケーション。
#       CSV/Excelインポート、属性別セグメント分析、共起ネットワーク、感情分析、N-gram、対応分析を搭載。
# Version: v2.0 (プロフェッショナル版)

import streamlit as st
import spacy
from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
import networkx as nx
import scipy.linalg

# --------------------------------------------------------------------------
# 1. PyInstaller対応ヘルパー関数 & 初期設定
# --------------------------------------------------------------------------

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
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

# --- Streamlit ページ設定 ---
st.set_page_config(layout="wide", page_title="プロフェッショナル・テキストマイニングツール")

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
    st.session_state.selected_pos = config.get("selected_pos", ['NOUN', 'PROPN', 'VERB', 'ADJ'])
    
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
            "selected_pos": st.session_state.selected_pos
        })
    
    st.session_state.initialized = True

def load_sentiment_dict():
    """ローカルの感情極性辞書をロードする"""
    # assets 内のパスを解決
    dict_path = resource_path('assets/sentiment_dict.csv')
    sent_dict = {}
    if os.path.exists(dict_path):
        try:
            df_sent = pd.read_csv(dict_path, encoding='utf-8')
            for _, row in df_sent.iterrows():
                sent_dict[str(row['word'])] = str(row['polarity'])
        except Exception as e:
            st.warning(f"感情極性辞書の読み込みに失敗しました: {e}")
    return sent_dict

# --------------------------------------------------------------------------
# 3. 解析エンジン・コア機能
# --------------------------------------------------------------------------

@st.cache_resource
def load_ginza_model():
    """GiNZAモデルをキャッシュして読み込む"""
    try:
        model_path = resource_path('ja_ginza')
        return spacy.load(model_path)
    except OSError:
        try:
            return spacy.load('ja_ginza')
        except OSError:
            st.error("GiNZAモデルが見つかりません。")
            st.stop()

nlp = load_ginza_model()

def perform_correspondence_analysis(df_tokens, df_sentences, attr_col, top_k=50):
    """SVDを用いて対応分析を計算する"""
    try:
        # トークンに属性値を結合
        df_token_attr = pd.merge(df_tokens, df_sentences[['sentence_id', 'attr_value']], on='sentence_id', how='left')
        
        # 頻出トップKの単語を抽出
        top_words = df_token_attr.groupby('word').size().sort_values(ascending=False).head(top_k).index.tolist()
        df_filtered = df_token_attr[df_token_attr['word'].isin(top_words)]
        
        if df_filtered.empty:
            return None
            
        # クロス集計表 (単語 × 属性)
        ct = pd.crosstab(df_filtered['word'], df_filtered['attr_value'])
        
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return None
            
        X = ct.values.astype(float)
        N = X.sum()
        if N == 0:
            return None
            
        P = X / N
        
        row_sums = P.sum(axis=1)
        col_sums = P.sum(axis=0)
        
        # 0割りの防止
        row_sums[row_sums == 0] = 1e-10
        col_sums[col_sums == 0] = 1e-10
        
        Dr_inv_sqrt = np.diag(1.0 / np.sqrt(row_sums))
        Dc_inv_sqrt = np.diag(1.0 / np.sqrt(col_sums))
        
        # 標準化残差
        rc = np.outer(row_sums, col_sums)
        S = Dr_inv_sqrt @ (P - rc) @ Dc_inv_sqrt
        
        # 特異値分解
        U, s, Vt = scipy.linalg.svd(S, full_matrices=False)
        
        if len(s) < 2:
            return None
            
        # 2次元座標の算出
        R = Dr_inv_sqrt @ U[:, :2] @ np.diag(s[:2])
        C = Dc_inv_sqrt @ Vt[:2, :].T @ np.diag(s[:2])
        
        df_rows = pd.DataFrame({
            'name': ct.index,
            'x': R[:, 0],
            'y': R[:, 1],
            'type': '単語'
        })
        
        df_cols = pd.DataFrame({
            'name': ct.columns,
            'x': C[:, 0],
            'y': C[:, 1],
            'type': '属性'
        })
        
        df_ca = pd.concat([df_rows, df_cols], ignore_index=True)
        return df_ca
    except Exception as e:
        st.warning(f"対応分析の実行中にエラーが発生しました: {e}")
        return None

def perform_full_analysis(df, text_col, attr_col, selected_pos, stop_words, import_type):
    """形態素解析、統計、頻度、TF-IDF、共起、感情、対応分析をまとめて計算する"""
    progress_bar = st.progress(0, text="解析の準備中...")
    
    sentiment_dict = load_sentiment_dict()
    
    all_sentences = []
    all_tokens = []
    
    raw_token_count = 0
    raw_unique_lemmas = set()
    
    total_rows = len(df)
    
    for row_idx, row in df.iterrows():
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        attr_val = str(row[attr_col]) if attr_col and pd.notna(row[attr_col]) else "未設定"
        
        # 文区切り (。 または 改行)
        sentences = [s.strip() for s in text.replace('\r\n', '\n').replace('\r', '\n').split('。') if s.strip()]
        if not sentences:
            sentences = [text.strip()] if text.strip() else []
            
        for sent_idx, sent in enumerate(sentences):
            doc = nlp(sent)
            sent_tokens = []
            pos_count = 0
            neg_count = 0
            
            for token in doc:
                # 前処理前の基礎統計計算用
                if not token.is_punct and not token.is_space and len(token.lemma_.strip()) > 0:
                    raw_token_count += 1
                    raw_unique_lemmas.add(token.lemma_)
                    
                    pos_tag = token.pos_
                    if pos_tag in ['CCONJ', 'SCONJ']:
                        pos_tag = 'CONJ'
                    
                    # 感情判定
                    word_lemma = token.lemma_
                    sentiment = sentiment_dict.get(word_lemma, None)
                    if sentiment == 'p':
                        pos_count += 1
                    elif sentiment == 'n':
                        neg_count += 1
                        
                    # ユーザー指定の品詞フィルタ & ストップワード
                    if pos_tag in selected_pos and word_lemma not in stop_words:
                        sent_tokens.append({
                            'word': word_lemma,
                            'pos': pos_tag,
                            'sentiment': sentiment
                        })
            
            # 文単位の感情スコア計算
            denom = pos_count + neg_count
            sent_score = (pos_count - neg_count) / denom if denom > 0 else 0.0
            if sent_score > 0.05:
                sent_class = 'ポジティブ'
            elif sent_score < -0.05:
                sent_class = 'ネガティブ'
            else:
                sent_class = 'ニュートラル'
                
            sentence_id = f"{row_idx}_{sent_idx}"
            all_sentences.append({
                'doc_id': row_idx,
                'sentence_id': sentence_id,
                'text': sent,
                'pos_count': pos_count,
                'neg_count': neg_count,
                'score': sent_score,
                'class': sent_class,
                'attr_value': attr_val
            })
            
            for t in sent_tokens:
                all_tokens.append({
                    'doc_id': row_idx,
                    'sentence_id': sentence_id,
                    'word': t['word'],
                    'pos': t['pos'],
                    'sentiment': t['sentiment']
                })
                
        progress_percentage = int((row_idx + 1) / total_rows * 70)
        progress_bar.progress(progress_percentage, text=f"形態素解析中... ({row_idx + 1}/{total_rows}行完了)")
        
    if not all_tokens:
        progress_bar.empty()
        st.warning("解析対象の単語が見つかりませんでした。テキストまたは除外設定を見直してください。")
        return None, None, None, None, None, None, None, None
        
    df_tokens = pd.DataFrame(all_tokens)
    df_sentences = pd.DataFrame(all_sentences)
    
    # 1. 頻度集計
    progress_bar.progress(75, text="単語頻度の集計中...")
    df_freq = df_tokens.groupby(['word', 'pos']).size().reset_index(name='出現回数').sort_values(by='出現回数', ascending=False)
    # 感情極性を結合
    pos_info_sent = df_tokens[['word', 'sentiment']].drop_duplicates(subset='word')
    df_freq = pd.merge(df_freq, pos_info_sent, on='word', how='left')
    
    # 2. TF-IDF集計
    progress_bar.progress(80, text="TF-IDF重要度の計算中...")
    doc_col = 'doc_id' if import_type == 'CSV / Excel ファイル' else 'sentence_id'
    doc_groups = df_tokens.groupby(doc_col)
    docs = [" ".join(group['word'].tolist()) for _, group in doc_groups]
    
    df_tfidf = pd.DataFrame()
    if len(docs) > 1:
        try:
            vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
            tfidf_matrix = vectorizer.fit_transform(docs)
            feature_names = vectorizer.get_feature_names_out()
            avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            df_tfidf = pd.DataFrame({'word': feature_names, '平均重要度スコア': avg_tfidf})
            
            pos_info = df_tokens[['word', 'pos', 'sentiment']].drop_duplicates(subset='word')
            df_tfidf = pd.merge(df_tfidf, pos_info, on='word', how='left').sort_values(by='平均重要度スコア', ascending=False)
        except Exception as e:
            st.warning(f"TF-IDFの計算中にエラーが発生しました: {e}")
            
    # 3. N-grams (Bigram / Trigram)
    progress_bar.progress(85, text="N-gram（連語）の抽出中...")
    bigrams = []
    trigrams = []
    for _, group in df_tokens.groupby('sentence_id'):
        words = group['word'].tolist()
        for i in range(len(words) - 1):
            bigrams.append(f"{words[i]} - {words[i+1]}")
        for i in range(len(words) - 2):
            trigrams.append(f"{words[i]} - {words[i+1]} - {words[i+2]}")
            
    df_bigrams = pd.DataFrame(Counter(bigrams).most_common(), columns=['連語', '出現回数'])
    df_bigrams['タイプ'] = 'Bigram (2語連語)'
    df_trigrams = pd.DataFrame(Counter(trigrams).most_common(), columns=['連語', '出現回数'])
    df_trigrams['タイプ'] = 'Trigram (3語連語)'
    df_ngrams = pd.concat([df_bigrams, df_trigrams], ignore_index=True)
    
    # 4. 共起関係の計算 (Jaccard係数)
    progress_bar.progress(90, text="共起関係の集計中...")
    top_words = df_freq.head(100)['word'].tolist()
    sent_groups = df_tokens.groupby('sentence_id')
    word_sets = [set(group[group['word'].isin(top_words)]['word']) for _, group in sent_groups]
    
    word_df = Counter()
    pair_df = Counter()
    for w_set in word_sets:
        for w in w_set:
            word_df[w] += 1
        w_list = list(w_set)
        for i in range(len(w_list)):
            for j in range(i + 1, len(w_list)):
                w1, w2 = sorted([w_list[i], w_list[j]])
                pair_df[(w1, w2)] += 1
                
    edges = []
    for (w1, w2), cooc_count in pair_df.items():
        df1 = word_df[w1]
        df2 = word_df[w2]
        jaccard = cooc_count / (df1 + df2 - cooc_count)
        edges.append({
            'word1': w1,
            'word2': w2,
            'cooc': cooc_count,
            'jaccard': jaccard
        })
    df_edges = pd.DataFrame(edges).sort_values(by='jaccard', ascending=False) if edges else pd.DataFrame(columns=['word1', 'word2', 'cooc', 'jaccard'])
    
    # 5. 対応分析 (SVD)
    df_ca = None
    if attr_col is not None:
        progress_bar.progress(95, text="対応分析のマップ作成中...")
        df_ca = perform_correspondence_analysis(df_tokens, df_sentences, attr_col)
        
    # コーパス統計
    ttr = len(raw_unique_lemmas) / raw_token_count if raw_token_count > 0 else 0.0
    avg_sent_len = raw_token_count / len(all_sentences) if all_sentences else 0.0
    corpus_stats = {
        '総文数': len(all_sentences),
        '総単語数 (前処理前)': raw_token_count,
        '異なり単語数 (前処理前)': len(raw_unique_lemmas),
        '語彙多様性指数 (TTR)': ttr,
        '平均文長 (単語数)': avg_sent_len
    }
    
    progress_bar.progress(100, text="すべての解析が完了しました！")
    time.sleep(0.5)
    progress_bar.empty()
    
    return df_freq, df_tfidf, df_ngrams, df_edges, df_sentences, df_tokens, df_ca, corpus_stats

# --------------------------------------------------------------------------
# 4. 可視化グラフ生成関数
# --------------------------------------------------------------------------

def generate_wordcloud_fig(frequencies, font_path):
    """ワードクラウド画像を生成する"""
    if not frequencies:
        return None
    
    font_path_to_use = None
    if font_path and os.path.exists(font_path):
        font_path_to_use = font_path
    else:
        try:
            bundled_font = resource_path('assets/NotoSansJP-Regular.ttf')
            if os.path.exists(bundled_font):
                font_path_to_use = bundled_font
        except Exception:
            pass

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
            st.warning("日本語フォントが見つからないため文字化けする可能性があります。")
            font_path_to_use = None

    try:
        wc = WordCloud(font_path=font_path_to_use, 
                       width=800, height=500, background_color='white', 
                       max_words=100, collocations=False, max_font_size=80
                       ).generate_from_frequencies(frequencies)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception as e:
        st.error(f"ワードクラウド生成エラー: {e}")
        return None

def generate_cooc_plotly(df_edges, df_freq, top_n_edges=40, layout_k=0.4):
    """Plotlyを用いたインタラクティブな共起ネットワーク"""
    if df_edges.empty:
        return None
        
    df_subset = df_edges.head(top_n_edges)
    
    G = nx.Graph()
    for _, row in df_subset.iterrows():
        G.add_edge(row['word1'], row['word2'], weight=row['jaccard'])
        
    # 力学モデルでノード配置を計算
    pos = nx.spring_layout(G, k=layout_k, seed=42)
    
    freq_dict = dict(zip(df_freq['word'], df_freq['出現回数']))
    
    # エッジ描画用の座標
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='rgba(150, 150, 150, 0.4)'),
        hoverinfo='none',
        mode='lines'
    )
    
    # ノード描画用の座標
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    degrees = dict(G.degree())
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node} (頻度: {freq_dict.get(node, 1)})")
        
        # 頻度に基づくノードサイズ
        size = 10 + np.sqrt(freq_dict.get(node, 1)) * 4
        node_size.append(size)
        node_color.append(degrees[node])
        
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n for n in G.nodes()],
        hovertext=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title=dict(text='接続数 (Degree)', side='right'),
                xanchor='left'
            ),
            line=dict(width=1.5, color='black')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0,l=0,r=0,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                 )
    )
    return fig

def generate_ca_plotly(df_ca):
    """対応分析結果の2次元散布図マップ"""
    if df_ca is None or df_ca.empty:
        return None
        
    fig = go.Figure()
    
    # 単語のプロット
    df_words = df_ca[df_ca['type'] == '単語']
    fig.add_trace(go.Scatter(
        x=df_words['x'],
        y=df_words['y'],
        mode='markers+text',
        name='単語',
        text=df_words['name'],
        textposition="top center",
        marker=dict(size=8, color='#1f77b4', opacity=0.7),
        hovertemplate='単語: %{text}<br>第一成分: %{x:.3f}<br>第二成分: %{y:.3f}<extra></extra>'
    ))
    
    # 属性カテゴリのプロット
    df_attrs = df_ca[df_ca['type'] == '属性']
    fig.add_trace(go.Scatter(
        x=df_attrs['x'],
        y=df_attrs['y'],
        mode='markers+text',
        name='属性カテゴリ',
        text=df_attrs['name'],
        textposition="top center",
        marker=dict(size=14, color='#ff7f0e', symbol='diamond', line=dict(width=2, color='black')),
        hovertemplate='属性: %{text}<br>第一成分: %{x:.3f}<br>第二成分: %{y:.3f}<extra></extra>'
    ))
    
    # 原点を通る補助線
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        xaxis_title='第一成分軸',
        yaxis_title='第二成分軸',
        hovermode='closest',
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

# --------------------------------------------------------------------------
# 5. UI操作用コールバック関数
# --------------------------------------------------------------------------

def set_analysis_flag_off():
    st.session_state.analysis_complete = False

def update_text_from_upload():
    uploaded_file = st.session_state.get('file_uploader_widget')
    if uploaded_file:
        encodings = ['utf-8', 'shift-jis', 'cp932']
        decoded = False
        for enc in encodings:
            try:
                st.session_state.input_text = uploaded_file.read().decode(enc)
                uploaded_file.seek(0)
                decoded = True
                break
            except UnicodeDecodeError:
                continue
        
        if decoded:
            set_analysis_flag_off()
        else:
            st.error("ファイルのデコードに失敗しました。(対応: utf-8, shift-jis, cp932)")

def handle_text_area_change():
    st.session_state.input_text = st.session_state.get('text_area_widget', '')
    set_analysis_flag_off()

def clear_input_text():
    st.session_state.input_text = ""
    set_analysis_flag_off()

def handle_file_upload():
    uploaded_file = st.session_state.get('csv_excel_uploader')
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                encodings = ['utf-8', 'shift-jis', 'cp932']
                for enc in encodings:
                    try:
                        uploaded_file.seek(0)
                        st.session_state.df_uploaded = pd.read_csv(uploaded_file, encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    st.error("CSVファイルのデコードに失敗しました。エンコーディングを確認してください。")
            else:
                st.session_state.df_uploaded = pd.read_excel(uploaded_file)
            set_analysis_flag_off()
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")

# --------------------------------------------------------------------------
# 6. Streamlit UI構築 (メイン処理)
# --------------------------------------------------------------------------

def main():
    initialize_session_state()

    st.title("🔬 プロフェッショナル・テキストマイニングツール")
    st.markdown("自然言語学的なコーパス分析と、マーケティング用ユーザーインサイト獲得のための高度な分析を実行します。")

    # --- サイドバー ---
    with st.sidebar:
        st.header("⚙️ 分析対象・フィルタ設定")
        
        # 品詞選択
        st.subheader("分析対象とする品詞")
        selected_pos_jp = st.multiselect(
            "品詞を選択 (複数可)",
            options=list(POS_MAP_JAPANESE.keys()),
            default=[k for k, v in POS_MAP_JAPANESE.items() if v in st.session_state.selected_pos],
            on_change=set_analysis_flag_off
        )
        st.session_state.selected_pos = [POS_MAP_JAPANESE[name] for name in selected_pos_jp]

        st.subheader("グラフ表示件数")
        st.session_state.top_n = st.slider(
            "表示する単語数", min_value=5, max_value=100, 
            value=st.session_state.top_n, on_change=set_analysis_flag_off
        )

        st.subheader("フォント設定")
        st.session_state.font_path = st.text_input(
            "カスタムフォントパス (.ttf / .ttc)", value=st.session_state.font_path,
            placeholder='例: /System/Library/Fonts/Hiragino Sans GB.ttc',
            on_change=set_analysis_flag_off,
            help="空欄の場合、システムの日本語フォントを自動検索します。"
        )
        
        st.header("🚫 除外ワード（ストップワード）")
        st.session_state.stop_words = [
            word.strip() for word in st.text_area(
                "除外リスト (1行に1つ)", value="\n".join(st.session_state.stop_words),
                height=180, on_change=set_analysis_flag_off
            ).split('\n') if word.strip()
        ]

    # --- メインコンテンツ：1. データのインポート ---
    st.header("1. データのインポート")
    st.session_state.import_type = st.radio(
        "インポート方法の選択",
        ["直接入力 / テキストファイル", "CSV / Excel ファイル"],
        index=0 if st.session_state.import_type == "直接入力 / テキストファイル" else 1,
        on_change=set_analysis_flag_off,
        horizontal=True
    )

    if st.session_state.import_type == "直接入力 / テキストファイル":
        st.file_uploader(
            "テキストファイル (.txt) をアップロード", type=['txt'],
            key='file_uploader_widget', on_change=update_text_from_upload
        )
        st.text_area(
            "または、ここにテキストを直接入力", value=st.session_state.input_text,
            key='text_area_widget', on_change=handle_text_area_change, height=180
        )
        if st.button("テキストをクリア", on_click=clear_input_text):
            st.rerun()
    else:
        st.file_uploader(
            "CSV または Excel ファイルをアップロード (.csv, .xlsx)", 
            type=['csv', 'xlsx'],
            key='csv_excel_uploader',
            on_change=handle_file_upload
        )
        
        if st.session_state.df_uploaded is not None:
            st.success(f"ファイルを読み込みました。 (総数: {len(st.session_state.df_uploaded)}行)")
            st.dataframe(st.session_state.df_uploaded.head(5), use_container_width=True)
            
            cols = list(st.session_state.df_uploaded.columns)
            st.session_state.text_col = st.selectbox(
                "分析対象のテキスト列", 
                options=cols,
                index=cols.index(st.session_state.text_col) if st.session_state.text_col in cols else 0,
                on_change=set_analysis_flag_off
            )
            
            attr_options = [None] + cols
            current_attr = st.session_state.attr_col
            index_attr = attr_options.index(current_attr) if current_attr in attr_options else 0
            
            st.session_state.attr_col = st.selectbox(
                "属性（セグメント）分類用の列 (任意)", 
                options=attr_options,
                index=index_attr,
                format_func=lambda x: "指定なし (全体分析)" if x is None else x,
                on_change=set_analysis_flag_off,
                help="満足度、年代、性別などのカテゴリ列を選択すると、属性別の分析が可能になります。"
            )

    # --- 2. 解析の実行 ---
    st.header("2. 解析の実行")
    if st.button("解析実行", type="primary"):
        df_to_analyze = None
        text_col_to_use = "text"
        attr_col_to_use = None
        
        if st.session_state.import_type == "直接入力 / テキストファイル":
            if st.session_state.input_text.strip():
                # 文でスプリットしてDataFrameにする
                raw_text = st.session_state.input_text
                sentences = [s.strip() for s in raw_text.replace('\r\n', '\n').replace('\r', '\n').split('。') if s.strip()]
                if not sentences:
                    sentences = [raw_text.strip()] if raw_text.strip() else []
                df_to_analyze = pd.DataFrame({"text": sentences})
                text_col_to_use = "text"
                attr_col_to_use = None
            else:
                st.error("テキストが入力されていません。")
        else:
            if st.session_state.df_uploaded is not None:
                df_to_analyze = st.session_state.df_uploaded
                text_col_to_use = st.session_state.text_col
                attr_col_to_use = st.session_state.attr_col
            else:
                st.error("ファイルがアップロードされていません。")
                
        if df_to_analyze is not None:
            df_freq, df_tfidf, df_ngrams, df_edges, df_sentences, df_tokens, df_ca, corpus_stats = perform_full_analysis(
                df_to_analyze, 
                text_col_to_use, 
                attr_col_to_use,
                st.session_state.selected_pos,
                st.session_state.stop_words,
                st.session_state.import_type
            )
            
            if df_freq is not None:
                st.session_state.df_freq = df_freq
                st.session_state.df_tfidf = df_tfidf if df_tfidf is not None else pd.DataFrame()
                st.session_state.df_ngrams = df_ngrams
                st.session_state.df_edges = df_edges
                st.session_state.df_sentences = df_sentences
                st.session_state.df_tokens = df_tokens
                st.session_state.df_ca = df_ca
                st.session_state.corpus_stats = corpus_stats
                st.session_state.analysis_complete = True
                
                # 設定保存
                save_config({
                    "top_n": st.session_state.top_n,
                    "font_path": st.session_state.font_path,
                    "stop_words": st.session_state.stop_words,
                    "selected_pos": st.session_state.selected_pos
                })
                st.toast("すべての解析が成功しました。結果を表示します。")
                st.rerun()

    # --- 3. 解析結果の表示 ---
    if st.session_state.analysis_complete:
        st.markdown("---")
        st.success("📊 解析結果")
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 統計 & 単語頻度", 
            "📈 重要度 (TF-IDF)", 
            "🔗 共起ネットワーク", 
            "🧩 N-gram（連語）分析", 
            "🎭 感情分析", 
            "🗺️ セグメント & 対応分析"
        ])

        # --- Tab 1: 統計 & 単語頻度 ---
        with tab1:
            st.subheader("📝 コーパス基本統計情報")
            stats = st.session_state.corpus_stats
            col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
            col_s1.metric("総文数", f"{stats['総文数']} 文")
            col_s2.metric("総単語数 (前処理前)", f"{stats['総単語数 (前処理前)']} 語")
            col_s3.metric("異なり単語数 (前処理前)", f"{stats['異なり単語数 (前処理前)']} 語")
            col_s4.metric("語彙多様性 (TTR)", f"{stats['語彙多様性指数 (TTR)']:.3f}")
            col_s5.metric("平均文長 (単語数)", f"{stats['平均文長 (単語数)']:.1f} 語")
            
            st.markdown("---")
            st.subheader("📊 単語頻出度")
            
            df_display = st.session_state.df_freq.copy()
            df_display['品詞'] = df_display['pos'].map(POS_MAP_JAPANESE_REV).fillna(df_display['pos'])
            
            if df_display.empty:
                st.warning("条件に合う単語が見つかりませんでした。")
            else:
                col1, col2 = st.columns([0.55, 0.45])
                with col1:
                    fig_bar = px.bar(
                        df_display.head(st.session_state.top_n), 
                        x='word', y='出現回数', color='品詞', text_auto=True,
                        title=f"頻出トップ {st.session_state.top_n} 単語"
                    ).update_layout(xaxis_title=None, yaxis_title="出現回数")
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    word_counts = dict(zip(df_display['word'], df_display['出現回数']))
                    fig_wc = generate_wordcloud_fig(word_counts, st.session_state.font_path)
                    if fig_wc:
                        st.pyplot(fig_wc)
                
                st.subheader("データテーブル")
                st.download_button("📥 頻度データをCSVで保存", df_display.to_csv(index=False, encoding='utf-8-sig'), f"word_freq_{timestamp}.csv", "text/csv")
                st.dataframe(df_display[['word', '品詞', '出現回数']], height=250, use_container_width=True)

        # --- Tab 2: TF-IDF ---
        with tab2:
            st.subheader("📈 重要度分析 (TF-IDF)")
            st.info("TF-IDFは、他の文や文書にあまり出現せず、その対象文書で特徴的に出現する単語に高い重要度スコアを与えます。")
            
            df_tfidf_display = st.session_state.df_tfidf.copy()
            if df_tfidf_display.empty:
                st.warning("重要度スコアを計算できる文書数が足りません（少なくとも2文または2行以上のデータが必要です）。")
            else:
                df_tfidf_display['品詞'] = df_tfidf_display['pos'].map(POS_MAP_JAPANESE_REV).fillna(df_tfidf_display['pos'])
                col1, col2 = st.columns([0.55, 0.45])
                with col1:
                    fig_bar_tfidf = px.bar(
                        df_tfidf_display.head(st.session_state.top_n),
                        x='word', y='平均重要度スコア', color='品詞', text_auto='.3f',
                        title=f"重要キーワードトップ {st.session_state.top_n}"
                    ).update_layout(xaxis_title=None)
                    st.plotly_chart(fig_bar_tfidf, use_container_width=True)
                    
                with col2:
                    tfidf_dict = dict(zip(df_tfidf_display['word'], df_tfidf_display['平均重要度スコア'] * 1000))
                    fig_wc_tfidf = generate_wordcloud_fig(tfidf_dict, st.session_state.font_path)
                    if fig_wc_tfidf:
                        st.pyplot(fig_wc_tfidf)
                        
                st.subheader("データテーブル")
                st.download_button("📥 TF-IDFデータをCSVで保存", df_tfidf_display.to_csv(index=False, encoding='utf-8-sig'), f"tfidf_{timestamp}.csv", "text/csv")
                st.dataframe(df_tfidf_display[['word', '品詞', '平均重要度スコア']], height=250, use_container_width=True)

        # --- Tab 3: 共起ネットワーク ---
        with tab3:
            st.subheader("🔗 単語共起ネットワーク")
            st.markdown("同一の文の中で同時に出現しやすい単語同士を、線の結びつき（ネットワーク）で表します。")
            
            df_edges = st.session_state.df_edges
            if df_edges.empty:
                st.warning("共起関係を検出できませんでした。")
            else:
                col_c1, col_c2 = st.columns([0.3, 0.7])
                with col_c1:
                    st.info("🟢 **ノードの大きさ**: 単語の頻度を表します。\n\n🔵 **ノードの色**: 接続数（他の単語との結びつきの多さ）を表します。")
                    top_n_edges = st.slider("表示する共起関係（エッジ）数", min_value=10, max_value=100, value=40)
                    layout_k = st.slider("ネットワークの広がり度合い (レイアウト係数)", min_value=0.1, max_value=1.5, value=0.4, step=0.1)
                
                with col_c2:
                    fig_cooc = generate_cooc_plotly(df_edges, st.session_state.df_freq, top_n_edges, layout_k)
                    if fig_cooc:
                        st.plotly_chart(fig_cooc, use_container_width=True)
                
                st.subheader("共起データ")
                st.download_button("📥 共起関係データをCSVで保存", df_edges.to_csv(index=False, encoding='utf-8-sig'), f"cooccurrence_{timestamp}.csv", "text/csv")
                st.dataframe(df_edges.rename(columns={'word1': '単語1', 'word2': '単語2', 'cooc': '共起回数', 'jaccard': '共起度 (Jaccard)'}), height=250, use_container_width=True)

        # --- Tab 4: N-gram ---
        with tab4:
            st.subheader("🧩 N-gram（連語）分析")
            st.markdown("テキスト内で隣り合って出現する2単語 (Bigram) または3単語 (Trigram) を集計します。フレーズの定型表現を発見するのに適しています。")
            
            df_ngrams = st.session_state.df_ngrams
            if df_ngrams.empty:
                st.warning("連語を抽出できませんでした。")
            else:
                ngram_type = st.radio("連語タイプの選択", ["Bigram (2語連語)", "Trigram (3語連語)"], horizontal=True)
                df_ng_filtered = df_ngrams[df_ngrams['タイプ'] == ngram_type].sort_values(by='出現回数', ascending=False)
                
                if df_ng_filtered.empty:
                    st.warning("該当するデータがありません。")
                else:
                    col1, col2 = st.columns([0.6, 0.4])
                    with col1:
                        fig_ngram = px.bar(
                            df_ng_filtered.head(st.session_state.top_n),
                            x='連語', y='出現回数', text_auto=True,
                            title=f"頻出 {ngram_type} トップ {st.session_state.top_n}"
                        ).update_layout(xaxis_title=None)
                        st.plotly_chart(fig_ngram, use_container_width=True)
                        
                    with col2:
                        st.subheader("データテーブル")
                        st.download_button("📥 連語データをCSVで保存", df_ng_filtered.to_csv(index=False, encoding='utf-8-sig'), f"ngrams_{timestamp}.csv", "text/csv")
                        st.dataframe(df_ng_filtered[['連語', '出現回数']], height=300, use_container_width=True)

        # --- Tab 5: 感情分析 ---
        with tab5:
            st.subheader("🎭 感情分析 (ポジネガ判定)")
            st.markdown("ローカルの評価極性辞書を用いて、各文または各行のポジティブ度・ネガティブ度を判定します。")
            
            df_sentences = st.session_state.df_sentences
            df_tokens = st.session_state.df_tokens
            
            col_e1, col_e2 = st.columns([0.4, 0.6])
            
            with col_e1:
                # 感情割合円グラフ
                sent_counts = df_sentences['class'].value_counts()
                fig_pie = px.pie(
                    names=sent_counts.index, 
                    values=sent_counts.values,
                    color=sent_counts.index,
                    color_discrete_map={'ポジティブ': '#2ca02c', 'ネガティブ': '#d62728', 'ニュートラル': '#7f7f7f'},
                    title='感情極性比率'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_e2:
                # 属性が指定されている場合、属性別の感情スコア平均値
                if st.session_state.attr_col is not None:
                    df_sent_by_attr = df_sentences.groupby('attr_value')['score'].mean().reset_index().sort_values(by='score', ascending=False)
                    fig_sent_bar = px.bar(
                        df_sent_by_attr,
                        x='attr_value', y='score',
                        color='score',
                        color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=0,
                        title='属性（セグメント）別 平均感情スコア',
                        labels={'score': '平均感情値 (-1: ネガ極大 ~ 1: ポジ極大)', 'attr_value': '属性カテゴリ'}
                    ).update_layout(xaxis_title=None)
                    st.plotly_chart(fig_sent_bar, use_container_width=True)
                else:
                    # 属性がない場合は平均感情スコアをメトリック表示
                    avg_score = df_sentences['score'].mean()
                    st.metric("全体の平均感情スコア (-1〜1)", f"{avg_score:.3f}", 
                              help="正の値はポジティブ、負の値はネガティブ寄りを示します。")
            
            st.markdown("---")
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                st.subheader("🟢 特徴的なポジティブ単語 (Top 15)")
                pos_words = df_tokens[df_tokens['sentiment'] == 'p']['word'].value_counts().head(15).reset_index(name='出現回数')
                if not pos_words.empty:
                    st.bar_chart(pos_words, x='word', y='出現回数', color='#2ca02c')
                else:
                    st.write("ポジティブ語が検出されませんでした。")
                    
            with col_w2:
                st.subheader("🔴 特徴的なネガティブ単語 (Top 15)")
                neg_words = df_tokens[df_tokens['sentiment'] == 'n']['word'].value_counts().head(15).reset_index(name='出現回数')
                if not neg_words.empty:
                    st.bar_chart(neg_words, x='word', y='出現回数', color='#d62728')
                else:
                    st.write("ネガティブ語が検出されませんでした。")
            
            st.markdown("---")
            st.subheader("データ詳細 (テキストと感情スコア)")
            st.download_button("📥 感情判定結果をCSVで保存", df_sentences.to_csv(index=False, encoding='utf-8-sig'), f"sentiment_analysis_{timestamp}.csv", "text/csv")
            st.dataframe(df_sentences[['text', 'class', 'score', 'attr_value']].rename(columns={'text': 'テキスト', 'class': '判定', 'score': 'スコア', 'attr_value': '属性'}), use_container_width=True)

        # --- Tab 6: セグメント & 対応分析 ---
        with tab6:
            st.subheader("🗺️ セグメント別クロス集計 & 対応分析 (コレスポンデンス分析)")
            
            if st.session_state.attr_col is None:
                st.warning("⚠️ 属性（セグメント）列が指定されていません。\n\n属性別の比較や対応分析を実行するには、インポートステップにおいてCSV/Excel形式のファイルをアップロードし、「属性（セグメント）分類用の列」を指定してください。")
            else:
                df_ca = st.session_state.df_ca
                if df_ca is None or df_ca.empty:
                    st.warning("データの分散が不足しているため、対応分析を構築できませんでした。")
                else:
                    st.markdown("対応分析は、カテゴリ（満足度や年代など）と特徴的な単語との距離を2次元マップ上に可視化します。")
                    st.info("💡 **見方**: 属性の点（オレンジダイヤ）の近くにある単語（ブルー）は、その属性で特に出現しやすい（親和性が高い）単語であることを示します。")
                    
                    fig_ca = generate_ca_plotly(df_ca)
                    if fig_ca:
                        st.plotly_chart(fig_ca, use_container_width=True)
                        
                    st.markdown("---")
                    st.subheader("属性別の単語クロス集計表")
                    
                    # クロス集計表を表示
                    df_token_attr = pd.merge(st.session_state.df_tokens, df_sentences[['sentence_id', 'attr_value']], on='sentence_id', how='left')
                    top_words_ct = df_token_attr.groupby('word').size().sort_values(ascending=False).head(st.session_state.top_n).index.tolist()
                    df_filtered_ct = df_token_attr[df_token_attr['word'].isin(top_words_ct)]
                    
                    if not df_filtered_ct.empty:
                        ct_table = pd.crosstab(df_filtered_ct['word'], df_filtered_ct['attr_value'])
                        st.download_button("📥 クロス集計表をCSVで保存", ct_table.to_csv(encoding='utf-8-sig'), f"cross_tabulation_{timestamp}.csv", "text/csv")
                        st.dataframe(ct_table, use_container_width=True)
                    else:
                        st.write("集計用データがありません。")

    elif 'initialized' in st.session_state and not st.session_state.analysis_complete:
        st.info("データを入力し、「解析実行」ボタンを押して分析を開始してください。")

if __name__ == "__main__":
    main()

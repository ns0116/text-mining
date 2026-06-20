# ファイル名: text_mining_app.py
# 説明: 自然言語学者およびプロのマーケター向けに高度化したテキストマイニングアプリケーション。
#       CSV/Excelインポート、属性別セグメント分析、共起ネットワーク、感情分析、N-gram、対応分析を搭載。
# Version: v2.0 (プロフェッショナル版) - リファクタリング済

import sys
import os
import streamlit as st
import pandas as pd
import datetime
import plotly.express as px

# プロジェクトルートディレクトリを python path に追加する
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Streamlit ページ設定は最初に実行する必要がある
st.set_page_config(layout="wide", page_title="プロフェッショナル・テキストマイニングツール")

# コアモジュールのインポート
from src.core.config import (
    POS_MAP_JAPANESE,
    POS_MAP_JAPANESE_REV,
    initialize_session_state,
    save_config
)
from src.core.stats import perform_full_analysis, run_nlp_morphology, perform_stats_analysis
from src.core.visualizer import (
    generate_wordcloud_fig,
    generate_cooc_plotly,
    generate_ca_plotly
)

# --------------------------------------------------------------------------
# UI操作用コールバック関数
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

def parse_synonyms_text(text):
    syn_dict = {}
    if not text:
        return syn_dict
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',') if p.strip()]
        if len(parts) >= 2:
            target = parts[0]
            for source in parts[1:]:
                syn_dict[source] = target
    return syn_dict


# --------------------------------------------------------------------------
# Streamlit UI構築 (メイン処理)
# --------------------------------------------------------------------------

def main():
    initialize_session_state()

    # --- インタラクティブ再計算（キャッシュされた生データがある場合） ---
    if st.session_state.raw_tokens is not None:
        try:
            synonyms_dict = parse_synonyms_text(st.session_state.synonyms_text)
            results = perform_stats_analysis(
                st.session_state.raw_tokens,
                st.session_state.raw_sentences,
                st.session_state.selected_pos,
                st.session_state.stop_words,
                synonyms_dict,
                st.session_state.import_type,
                st.session_state.attr_col,
                top_k=st.session_state.top_n
            )
            df_freq, df_tfidf, df_ngrams, df_edges, df_sentences, df_tokens, df_ca, corpus_stats = results
            
            st.session_state.df_freq = df_freq
            st.session_state.df_tfidf = df_tfidf if df_tfidf is not None else pd.DataFrame()
            st.session_state.df_ngrams = df_ngrams
            st.session_state.df_edges = df_edges
            st.session_state.df_sentences = df_sentences
            st.session_state.df_tokens = df_tokens
            st.session_state.df_ca = df_ca
            st.session_state.corpus_stats = corpus_stats
            st.session_state.analysis_complete = True
        except Exception as e:
            st.error(f"統計のリアルタイム更新中にエラーが発生しました: {e}")

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
        font_input = st.text_input(
            "カスタムフォントパス (.ttf / .ttc)", value=st.session_state.font_path,
            placeholder='例: /System/Library/Fonts/Hiragino Sans GB.ttc',
            help="空欄の場合、システムの日本語フォントを自動検索します。"
        )
        if font_input:
            if ".." in font_input or not (font_input.lower().endswith('.ttf') or font_input.lower().endswith('.ttc')):
                st.error("❌ セキュリティ警告: パストラバーサル（..）や、.ttf / .ttc 以外の拡張子は指定できません。")
                font_input = ""
        st.session_state.font_path = font_input

        st.subheader("🔄 類義語（シノニム）統合")
        st.session_state.synonyms_text = st.text_area(
            "代表語, 類義語1, 類義語2... (1行に1組)",
            value=st.session_state.synonyms_text,
            height=120,
            help="例: スマートフォン, スマホ, 携帯"
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
            # 進行バーをStreamlit側で生成し、コールバック経由で更新する
            progress_bar = st.progress(0, text="解析の準備中...")
            
            def streamlit_progress_callback(percentage, text):
                progress_bar.progress(percentage, text=text)

            try:
                df_raw_tokens, df_raw_sentences = run_nlp_morphology(
                    df_to_analyze, 
                    text_col_to_use, 
                    attr_col_to_use,
                    progress_callback=streamlit_progress_callback
                )
                
                st.session_state.raw_tokens = df_raw_tokens
                st.session_state.raw_sentences = df_raw_sentences
                st.session_state.text_col = text_col_to_use
                st.session_state.attr_col = attr_col_to_use
                
                synonyms_dict = parse_synonyms_text(st.session_state.synonyms_text)
                
                results = perform_stats_analysis(
                    df_raw_tokens,
                    df_raw_sentences,
                    st.session_state.selected_pos,
                    st.session_state.stop_words,
                    synonyms_dict,
                    st.session_state.import_type,
                    attr_col_to_use,
                    top_k=st.session_state.top_n
                )
                progress_bar.empty()
                
                df_freq, df_tfidf, df_ngrams, df_edges, df_sentences, df_tokens, df_ca, corpus_stats = results
                
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
                    "selected_pos": st.session_state.selected_pos,
                    "synonyms_text": st.session_state.synonyms_text
                })
                st.toast("すべての解析が成功しました。結果を表示します。")
                st.rerun()
            except ValueError as e:
                progress_bar.empty()
                st.warning(str(e))
            except RuntimeError as e:
                progress_bar.empty()
                st.error(str(e))
            except Exception as e:
                progress_bar.empty()
                st.error(f"解析中に想定外のエラーが発生しました: {e}")

    # --- 3. 解析結果の表示 ---
    if st.session_state.analysis_complete:
        st.markdown("---")
        st.success("📊 解析結果")
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 統計 & 単語頻度", 
            "📈 重要度 (TF-IDF)", 
            "🔗 共起ネットワーク", 
            "🧩 N-gram（連語）分析", 
            "🎭 感情分析", 
            "🗺️ セグメント & 対応分析",
            "🔍 原文ドリルダウン"
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
                        import matplotlib.pyplot as plt
                        plt.close(fig_wc)
                
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
                        import matplotlib.pyplot as plt
                        plt.close(fig_wc_tfidf)
                        
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
            st.markdown("ローカルの感情極性辞書を用いて、各文または各行のポジティブ度・ネガティブ度を判定します。")
            
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
                # カテゴリ数チェック
                unique_attrs = df_sentences['attr_value'].dropna().unique()
                if len(unique_attrs) == 2:
                    st.warning("⚠️ 注意: 属性のカテゴリ数が2つの場合（例：男性と女性のみ）、対応分析は数学的に1次元の軸に縮退します。2次元マップ上の第二成分軸（Y軸）はノイズデータであるため、横方向（第一成分軸）のみに着目してください。より信頼性の高い分析には、3つ以上のカテゴリ（例：満足、普通、不満）を持つ属性を使用することをお勧めします。")
                
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

        # --- Tab 7: 原文ドリルダウン ---
        with tab7:
            st.subheader("🔍 原文ドリルダウン (生の声検索)")
            st.markdown("特定のキーワードが含まれる文を検索し、文脈と属性・感情スコアを確認します。")
            
            search_word = st.text_input("検索するキーワードを入力", value="", placeholder="例: フリーズ")
            if search_word:
                # Filter sentences containing the search word
                df_sentences_drill = st.session_state.df_sentences[
                    st.session_state.df_sentences['text'].str.contains(search_word, case=False, na=False)
                ].copy()
                
                if df_sentences_drill.empty:
                    st.info(f"「{search_word}」を含む文は見つかりませんでした。")
                else:
                    st.write(f"該当件数: {len(df_sentences_drill)} 件")
                    
                    # Highlight word using HTML
                    import re
                    def highlight_text(txt):
                        escaped = re.escape(search_word)
                        try:
                            pattern = re.compile(f"({escaped})", re.IGNORECASE)
                            return pattern.sub(r"<mark style='background-color: #ffff00; color: black; padding: 2px;'>\1</mark>", txt)
                        except Exception:
                            return txt
                            
                    df_sentences_drill['ハイライトテキスト'] = df_sentences_drill['text'].apply(highlight_text)
                    
                    # Select and rename columns
                    df_show = df_sentences_drill[['ハイライトテキスト', 'attr_value', 'class', 'score']].rename(columns={
                        'ハイライトテキスト': 'テキスト',
                        'attr_value': '属性',
                        'class': '感情分類',
                        'score': '感情スコア'
                    })
                    
                    st.write(df_show.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.info("キーワードを入力すると、該当する文がここに表示されます。")

    elif 'initialized' in st.session_state and not st.session_state.analysis_complete:
        st.info("データを入力し、「解析実行」ボタンを押して分析を開始してください。")

if __name__ == "__main__":
    main()

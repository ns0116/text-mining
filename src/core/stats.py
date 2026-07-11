import time
import logging
import numpy as np
import pandas as pd
from collections import Counter
import scipy.linalg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.core.nlp_engine import load_ginza_model, load_sentiment_dict

logger = logging.getLogger(__name__)

def perform_correspondence_analysis(df_tokens, df_sentences, attr_col, top_k=50):
    """SVDを用いて対応分析を計算する"""
    try:
        if df_tokens.empty or df_sentences.empty or not attr_col:
            return None, [0.0, 0.0]

        # トークンに属性値を結合
        df_token_attr = pd.merge(df_tokens, df_sentences[['sentence_id', 'attr_value']], on='sentence_id', how='left')

        # 頻出トップKの単語を抽出
        top_words = df_token_attr.groupby('word').size().sort_values(ascending=False).head(top_k).index.tolist()
        df_filtered = df_token_attr[df_token_attr['word'].isin(top_words)]

        if df_filtered.empty:
            return None, [0.0, 0.0]

        # クロス集計表 (単語 × 属性)
        ct = pd.crosstab(df_filtered['word'], df_filtered['attr_value'])

        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return None, [0.0, 0.0]

        X = ct.values.astype(float)
        N = X.sum()
        if N == 0:
            return None, [0.0, 0.0]
            
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
            return None, [0.0, 0.0]

        # 寄与率の計算
        total_inertia = np.sum(s**2)
        variance_explained = (s**2 / total_inertia * 100)[:2].tolist() if total_inertia > 0 else [0.0, 0.0]

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
        return df_ca, variance_explained
    except Exception as e:
        logger.warning(f"対応分析の実行中にエラーが発生しました: {e}")
        return None, [0.0, 0.0]

def run_nlp_morphology(df, text_col, attr_col, progress_callback=None):
    """形態素解析処理のみを行い、生トークンと生文データを抽出して返す（重い処理）"""
    if progress_callback:
        progress_callback(0, "形態素解析の準備中...")
        
    nlp = load_ginza_model()
    sentiment_dict = load_sentiment_dict()
    
    all_sentences = []
    all_tokens = []
    
    total_rows = len(df)
    
    if hasattr(nlp, 'pipe'):
        # 実際のGiNZAモデル向け：チャンク分割バッチ処理
        texts = df[text_col].fillna("").astype(str).tolist()
        attrs = df[attr_col].fillna("未設定").astype(str).tolist() if attr_col else ["未設定"] * total_rows

        BATCH_SIZE = 100
        for chunk_start in range(0, total_rows, BATCH_SIZE):
            chunk_texts = texts[chunk_start:chunk_start + BATCH_SIZE]
            chunk_attrs = attrs[chunk_start:chunk_start + BATCH_SIZE]
            docs = nlp.pipe(chunk_texts, batch_size=32)
            for chunk_idx, doc in enumerate(docs):
                doc_idx = chunk_start + chunk_idx
                attr_val = chunk_attrs[chunk_idx]

                # spaCyによる文境界検出
                sents = list(doc.sents) if hasattr(doc, 'sents') else [doc]
                for sent_idx, sent in enumerate(sents):
                    sentence_id = f"{doc_idx}_{sent_idx}"
                    sent_text = sent.text.strip() if hasattr(sent, 'text') else str(sent).strip()

                    all_sentences.append({
                        'doc_id': doc_idx,
                        'sentence_id': sentence_id,
                        'text': sent_text,
                        'attr_value': attr_val
                    })

                    for token in sent:
                        is_punct = token.is_punct
                        is_space = token.is_space
                        lemma = token.lemma_

                        if not is_punct and not is_space and len(lemma.strip()) > 0:
                            pos_tag = token.pos_
                            if pos_tag in ['CCONJ', 'SCONJ']:
                                pos_tag = 'CONJ'

                            sentiment = sentiment_dict.get(lemma, None)

                            is_negated = False
                            if hasattr(token, 'children') and token.children:
                                for child in token.children:
                                    child_dep = child.dep_ if hasattr(child, 'dep_') else ''
                                    child_lemma = child.lemma_ if hasattr(child, 'lemma_') else str(child)
                                    if child_dep in ['aux', 'fixed', 'advcl'] and child_lemma in ['ない', 'ぬ', 'ず', 'かねる', 'せん', '無い', 'なし']:
                                        is_negated = True
                                        break

                            all_tokens.append({
                                'doc_id': doc_idx,
                                'sentence_id': sentence_id,
                                'word': lemma,
                                'pos': pos_tag,
                                'sentiment': sentiment,
                                'is_negated': is_negated
                            })
            if progress_callback:
                done = min(chunk_start + BATCH_SIZE, total_rows)
                progress_percentage = int(done / total_rows * 90)
                progress_callback(progress_percentage, f"形態素解析中... ({done}/{total_rows}行完了)")
    else:
        # モック/テスト用：1行・1文ずつの処理（モック対象が sentence_map に含まれるようにするため）
        for row_idx, row in df.iterrows():
            text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            attr_val = str(row[attr_col]) if attr_col and pd.notna(row[attr_col]) else "未設定"
            
            # 文区切り (。 または 改行)
            sentences = [s.strip() for s in text.replace('\r\n', '\n').replace('\r', '\n').split('。') if s.strip()]
            if not sentences:
                sentences = [text.strip()] if text.strip() else []
                
            for sent_idx, sent_text in enumerate(sentences):
                doc = nlp(sent_text)
                sentence_id = f"{row_idx}_{sent_idx}"
                all_sentences.append({
                    'doc_id': row_idx,
                    'sentence_id': sentence_id,
                    'text': sent_text,
                    'attr_value': attr_val
                })
                
                tokens_in_sent = list(doc) if hasattr(doc, '__iter__') else [doc]
                # MockDocがdocそのもので、tokensメンバを持つ場合
                if hasattr(doc, 'tokens'):
                    tokens_in_sent = doc.tokens
                    
                for token in tokens_in_sent:
                    is_punct = token.is_punct if hasattr(token, 'is_punct') else False
                    is_space = token.is_space if hasattr(token, 'is_space') else False
                    lemma = token.lemma_ if hasattr(token, 'lemma_') else str(token)
                    
                    if not is_punct and hasattr(token, 'text') and token.text in ['。', '、', '！', '？', '!', '?', ' ', '\n', '\t']:
                        is_punct = True
                        
                    if not is_punct and not is_space and len(lemma.strip()) > 0:
                        pos_tag = token.pos_ if hasattr(token, 'pos_') else 'NOUN'
                        if pos_tag in ['CCONJ', 'SCONJ']:
                            pos_tag = 'CONJ'
                            
                        sentiment = sentiment_dict.get(lemma, None)
                        
                        is_negated = False
                        if hasattr(token, 'children') and token.children:
                            for child in token.children:
                                child_dep = child.dep_ if hasattr(child, 'dep_') else ''
                                child_lemma = child.lemma_ if hasattr(child, 'lemma_') else str(child)
                                if child_dep in ['aux', 'fixed', 'advcl'] and child_lemma in ['ない', 'ぬ', 'ず', 'かねる', 'せん', '無い', 'なし']:
                                    is_negated = True
                                    break
                                    
                        all_tokens.append({
                            'doc_id': row_idx,
                            'sentence_id': sentence_id,
                            'word': lemma,
                            'pos': pos_tag,
                            'sentiment': sentiment,
                            'is_negated': is_negated
                        })
            if progress_callback:
                progress_percentage = int((row_idx + 1) / total_rows * 90)
                progress_callback(progress_percentage, f"形態素解析中... ({row_idx + 1}/{total_rows}行完了)")
                
    if progress_callback:
        progress_callback(100, "形態素解析が完了しました。")
        
    df_raw_tokens = pd.DataFrame(all_tokens) if all_tokens else pd.DataFrame(columns=['doc_id', 'sentence_id', 'word', 'pos', 'sentiment', 'is_negated'])
    df_raw_sentences = pd.DataFrame(all_sentences) if all_sentences else pd.DataFrame(columns=['doc_id', 'sentence_id', 'text', 'attr_value'])
    
    return df_raw_tokens, df_raw_sentences

def calculate_corpus_stats(df_raw_tokens, df_raw_sentences):
    """生データに基づき、フィルタリング前のコーパス基本統計を計算する"""
    if df_raw_tokens.empty:
        return {
            '総文数': len(df_raw_sentences),
            '総単語数 (前処理前)': 0,
            '異なり単語数 (前処理前)': 0,
            '語彙多様性指数 (TTR)': 0.0,
            '平均文長 (単語数)': 0.0
        }
        
    raw_token_count = len(df_raw_tokens)
    raw_unique_lemmas = set(df_raw_tokens['word'].tolist())
    ttr = len(raw_unique_lemmas) / raw_token_count if raw_token_count > 0 else 0.0
    avg_sent_len = raw_token_count / len(df_raw_sentences) if not df_raw_sentences.empty else 0.0
    
    return {
        '総文数': len(df_raw_sentences),
        '総単語数 (前処理前)': raw_token_count,
        '異なり単語数 (前処理前)': len(raw_unique_lemmas),
        '語彙多様性指数 (TTR)': ttr,
        '平均文長 (単語数)': avg_sent_len
    }

def perform_stats_analysis(df_raw_tokens, df_raw_sentences, selected_pos, stop_words, synonyms_dict, import_type, attr_col, top_k=50, document_resolution="行単位（文単位）", sentiment_threshold=0.05, min_ngram_count=1):
    """形態素解析済みのデータに対して、品詞・除外・表記揺れのフィルタリングを適用し、各種統計結果を計算する（超高速処理）"""
    if df_raw_tokens.empty:
        raise ValueError("解析対象の単語が見つかりませんでした。テキストまたは除外設定を見直してください。")
        
    df_tokens = df_raw_tokens.copy()
    
    # 0. 類義語（シノニム）置換処理の適用
    if synonyms_dict:
        df_tokens['word'] = df_tokens['word'].replace(synonyms_dict)
        
    # アルファベットの大文字小文字や全角半角の標準化
    df_tokens['word_lower'] = df_tokens['word'].str.lower()
    
    # 感情極性の解決（否定語フラグがTrueなら極性を反転させる）
    df_tokens['resolved_sentiment'] = df_tokens['sentiment']
    neg_mask = df_tokens['is_negated'] == True
    df_tokens.loc[neg_mask & (df_tokens['sentiment'] == 'p'), 'resolved_sentiment'] = 'n'
    df_tokens.loc[neg_mask & (df_tokens['sentiment'] == 'n'), 'resolved_sentiment'] = 'p'
    
    # 除外ワード（ストップワード）の小文字化
    stop_words_lower = [str(w).strip().lower() for w in stop_words]
    
    # 1. フィルタ適用済みトークンリストの作成
    df_filtered_tokens = df_tokens[
        df_tokens['pos'].isin(selected_pos) & 
        (~df_tokens['word_lower'].isin(stop_words_lower))
    ]
    
    if df_filtered_tokens.empty:
        raise ValueError("解析対象の単語が見つかりませんでした。テキストまたは除外設定を見直してください。")
        
    # 1.1 頻度集計
    df_freq = df_filtered_tokens.groupby(['word', 'pos']).size().reset_index(name='出現回数').sort_values(by='出現回数', ascending=False)
    # 感情極性を結合
    pos_info_sent = df_tokens[['word', 'resolved_sentiment']].rename(columns={'resolved_sentiment': 'sentiment'}).drop_duplicates(subset='word')
    df_freq = pd.merge(df_freq, pos_info_sent, on='word', how='left')
    
    # 2. TF-IDF集計 (ダミーアナライザーを適用し、再分割を防止)
    df_tfidf = pd.DataFrame()
    try:
        if document_resolution == "属性グループ単位" and attr_col is not None:
            # 属性グループ単位での文書化
            df_token_attr = pd.merge(df_filtered_tokens, df_raw_sentences[['sentence_id', 'attr_value']], on='sentence_id', how='left')
            doc_groups = df_token_attr.groupby('attr_value')
            docs = [group['word'].tolist() for _, group in doc_groups]
        else:
            doc_col = 'doc_id' if import_type == 'CSV / Excel ファイル' else 'sentence_id'
            doc_groups = df_filtered_tokens.groupby(doc_col)
            docs = [group['word'].tolist() for _, group in doc_groups]

        if len(docs) > 1:
            # analyzer=lambda x: x とすることで、リストされたトークンをそのまま処理
            vectorizer = TfidfVectorizer(analyzer=lambda x: x, lowercase=False)
            tfidf_matrix = vectorizer.fit_transform(docs)
            feature_names = vectorizer.get_feature_names_out()
            avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            df_tfidf = pd.DataFrame({'word': feature_names, '平均重要度スコア': avg_tfidf})
            
            pos_info = df_filtered_tokens[['word', 'pos', 'resolved_sentiment']].rename(columns={'resolved_sentiment': 'sentiment'}).drop_duplicates(subset='word')
            df_tfidf = pd.merge(df_tfidf, pos_info, on='word', how='left').sort_values(by='平均重要度スコア', ascending=False)
    except Exception as e:
        logger.warning(f"TF-IDFの計算中にエラーが発生しました: {e}")
            
    # 3. N-grams (Bigram / Trigram) (品詞フィルタ「前」の系列から算出、その後構成要素が有効かをチェック)
    bigrams = []
    trigrams = []
    
    # N-gramは、文の中の元の並び順を利用して、両方の単語がフィルタ対象をクリアしている場合のみ抽出
    for _, group in df_tokens.groupby('sentence_id'):
        words = group['word'].tolist()
        poses = group['pos'].tolist()
        words_lower = group['word_lower'].tolist()
        
        # Bigram
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            p1, p2 = poses[i], poses[i+1]
            wl1, wl2 = words_lower[i], words_lower[i+1]
            if (p1 in selected_pos and wl1 not in stop_words_lower and
                p2 in selected_pos and wl2 not in stop_words_lower):
                bigrams.append(f"{w1} - {w2}")
                
        # Trigram
        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i+1], words[i+2]
            p1, p2, p3 = poses[i], poses[i+1], poses[i+2]
            wl1, wl2, wl3 = words_lower[i], words_lower[i+1], words_lower[i+2]
            if (p1 in selected_pos and wl1 not in stop_words_lower and
                p2 in selected_pos and wl2 not in stop_words_lower and
                p3 in selected_pos and wl3 not in stop_words_lower):
                trigrams.append(f"{w1} - {w2} - {w3}")
                
    df_bigrams = pd.DataFrame(
        [(k, v) for k, v in Counter(bigrams).items() if v >= min_ngram_count],
        columns=['連語', '出現回数']
    ).sort_values('出現回数', ascending=False)
    df_bigrams['タイプ'] = 'Bigram (2語連語)'
    df_trigrams = pd.DataFrame(
        [(k, v) for k, v in Counter(trigrams).items() if v >= min_ngram_count],
        columns=['連語', '出現回数']
    ).sort_values('出現回数', ascending=False)
    df_ngrams = pd.concat([df_bigrams, df_trigrams], ignore_index=True)
    
    # 4. 共起関係の計算 (CountVectorizerを用いた高速ベクトル積演算)
    top_words = df_freq.groupby('word')['出現回数'].sum().sort_values(ascending=False).head(100).index.tolist()
    if len(top_words) < 2:
        df_edges = pd.DataFrame(columns=['word1', 'word2', 'cooc', 'jaccard'])
    else:
        try:
            # 各文における top_words の出現状況を二値ベクトル化
            sent_word_lists = df_filtered_tokens[df_filtered_tokens['word'].isin(top_words)].groupby('sentence_id')['word'].apply(list).tolist()
            
            vectorizer = CountVectorizer(analyzer=lambda x: x, vocabulary=top_words, binary=True)
            X = vectorizer.fit_transform(sent_word_lists)
            
            # 行列積による共起回数算出
            C = (X.T * X).toarray()
            dfs = C.diagonal()
            
            edges = []
            for i in range(len(top_words)):
                for j in range(i + 1, len(top_words)):
                    cooc_count = C[i, j]
                    if cooc_count > 0:
                        df1 = dfs[i]
                        df2 = dfs[j]
                        jaccard = cooc_count / (df1 + df2 - cooc_count)
                        edges.append({
                            'word1': top_words[i],
                            'word2': top_words[j],
                            'cooc': cooc_count,
                            'jaccard': jaccard
                        })
            df_edges = pd.DataFrame(edges).sort_values(by='jaccard', ascending=False) if edges else pd.DataFrame(columns=['word1', 'word2', 'cooc', 'jaccard'])
        except Exception as e:
            logger.warning(f"共起分析の計算中にエラーが発生しました: {e}")
            df_edges = pd.DataFrame(columns=['word1', 'word2', 'cooc', 'jaccard'])
            
    # 5. 文単位の感情スコア更新
    sent_sentiment = df_tokens.groupby('sentence_id').agg(
        pos_count=('resolved_sentiment', lambda s: (s == 'p').sum()),
        neg_count=('resolved_sentiment', lambda s: (s == 'n').sum())
    ).reset_index()
    
    sent_sentiment['denom'] = sent_sentiment['pos_count'] + sent_sentiment['neg_count']
    sent_sentiment['score'] = np.where(
        sent_sentiment['denom'] > 0,
        (sent_sentiment['pos_count'] - sent_sentiment['neg_count']) / sent_sentiment['denom'],
        0.0
    )
    sent_sentiment['class'] = np.select(
        [sent_sentiment['score'] > sentiment_threshold, sent_sentiment['score'] < -sentiment_threshold],
        ['ポジティブ', 'ネガティブ'],
        default='ニュートラル'
    )
    
    df_sentences = pd.merge(df_raw_sentences, sent_sentiment[['sentence_id', 'pos_count', 'neg_count', 'score', 'class']], on='sentence_id', how='left')
    
    # 6. 対応分析 (SVD)
    df_ca = None
    ca_variance_explained = [0.0, 0.0]
    if attr_col is not None:
        df_ca, ca_variance_explained = perform_correspondence_analysis(df_filtered_tokens, df_sentences, attr_col, top_k)

    # コーパス統計
    corpus_stats = calculate_corpus_stats(df_raw_tokens, df_raw_sentences)
    corpus_stats['ca_variance_explained'] = ca_variance_explained
    
    return df_freq, df_tfidf, df_ngrams, df_edges, df_sentences, df_filtered_tokens, df_ca, corpus_stats

def perform_full_analysis(df, text_col, attr_col, selected_pos, stop_words, import_type, progress_callback=None, synonyms_dict=None):
    """互換性維持のためのラッパー関数"""
    if synonyms_dict is None:
        synonyms_dict = {}
    df_raw_tokens, df_raw_sentences = run_nlp_morphology(df, text_col, attr_col, progress_callback)
    return perform_stats_analysis(df_raw_tokens, df_raw_sentences, selected_pos, stop_words, synonyms_dict, import_type, attr_col)

def export_analysis_to_excel(df_freq, df_tfidf, df_ngrams, df_edges, df_sentences, corpus_stats):
    """すべての分析結果を個別のシートにまとめたExcelファイルを生成する"""
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: コーパス統計情報
        df_stats = pd.DataFrame(list(corpus_stats.items()), columns=['項目', '数値'])
        df_stats.to_excel(writer, sheet_name='基本統計', index=False)
        
        # Sheet 2: 単語頻度
        if df_freq is not None and not df_freq.empty:
            df_freq.to_excel(writer, sheet_name='単語頻度', index=False)
            
        # Sheet 3: 重要度 (TF-IDF)
        if df_tfidf is not None and not df_tfidf.empty:
            df_tfidf.to_excel(writer, sheet_name='重要度(TF-IDF)', index=False)
            
        # Sheet 4: N-grams（連語）
        if df_ngrams is not None and not df_ngrams.empty:
            df_ngrams.to_excel(writer, sheet_name='連語分析(N-gram)', index=False)
            
        # Sheet 5: 共起関係
        if df_edges is not None and not df_edges.empty:
            df_edges.to_excel(writer, sheet_name='共起関係', index=False)
            
        # Sheet 6: 原文 & 感情スコア
        if df_sentences is not None and not df_sentences.empty:
            df_sentences.to_excel(writer, sheet_name='原文と感情分析', index=False)
            
    output.seek(0)
    return output.getvalue()


import time
import logging
import numpy as np
import pandas as pd
from collections import Counter
import scipy.linalg
from sklearn.feature_extraction.text import TfidfVectorizer
from src.core.nlp_engine import load_ginza_model, load_sentiment_dict

logger = logging.getLogger(__name__)

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
        logger.warning(f"対応分析の実行中にエラーが発生しました: {e}")
        return None

def perform_full_analysis(df, text_col, attr_col, selected_pos, stop_words, import_type, progress_callback=None):
    """形態素解析、統計、頻度、TF-IDF、共起、感情、対応分析をまとめて計算する"""
    if progress_callback:
        progress_callback(0, "解析の準備中...")
    
    nlp = load_ginza_model()
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
                
        if progress_callback:
            progress_percentage = int((row_idx + 1) / total_rows * 70)
            progress_callback(progress_percentage, f"形態素解析中... ({row_idx + 1}/{total_rows}行完了)")
        
    if not all_tokens:
        raise ValueError("解析対象の単語が見つかりませんでした。テキストまたは除外設定を見直してください。")
        
    df_tokens = pd.DataFrame(all_tokens)
    df_sentences = pd.DataFrame(all_sentences)
    
    # 1. 頻度集計
    if progress_callback:
        progress_callback(75, "単語頻度の集計中...")
    df_freq = df_tokens.groupby(['word', 'pos']).size().reset_index(name='出現回数').sort_values(by='出現回数', ascending=False)
    # 感情極性を結合
    pos_info_sent = df_tokens[['word', 'sentiment']].drop_duplicates(subset='word')
    df_freq = pd.merge(df_freq, pos_info_sent, on='word', how='left')
    
    # 2. TF-IDF集計
    if progress_callback:
        progress_callback(80, "TF-IDF重要度の計算中...")
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
            logger.warning(f"TF-IDFの計算中にエラーが発生しました: {e}")
            
    # 3. N-grams (Bigram / Trigram)
    if progress_callback:
        progress_callback(85, "N-gram（連語）の抽出中...")
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
    if progress_callback:
        progress_callback(90, "共起関係の集計中...")
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
        if progress_callback:
            progress_callback(95, "対応分析のマップ作成中...")
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
    
    if progress_callback:
        progress_callback(100, "すべての解析が完了しました！")
    time.sleep(0.5)
    
    return df_freq, df_tfidf, df_ngrams, df_edges, df_sentences, df_tokens, df_ca, corpus_stats

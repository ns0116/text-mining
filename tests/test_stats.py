import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.core.stats import perform_correspondence_analysis, perform_full_analysis

# --- Helper Classes for Mocking spaCy ---
class MockToken:
    def __init__(self, text, lemma, pos, is_punct=False, is_space=False):
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
        self.is_punct = is_punct
        self.is_space = is_space

class MockDoc:
    def __init__(self, tokens):
        self.tokens = tokens
        
    def __iter__(self):
        return iter(self.tokens)

class MockNLP:
    def __init__(self, sentence_map):
        self.sentence_map = sentence_map
        
    def __call__(self, text):
        if text in self.sentence_map:
            return MockDoc(self.sentence_map[text])
        # Fallback: parse by space and treat as noun
        tokens = [MockToken(word, word, 'NOUN') for word in text.split()]
        return MockDoc(tokens)

# --- Tests for perform_correspondence_analysis ---

def test_perform_correspondence_analysis_success():
    # 2x2 table: 2 distinct words, 2 distinct attributes
    df_tokens = pd.DataFrame([
        {'word': '猫', 'sentence_id': 's1'},
        {'word': '猫', 'sentence_id': 's2'},
        {'word': '犬', 'sentence_id': 's3'},
        {'word': '犬', 'sentence_id': 's4'},
    ])
    df_sentences = pd.DataFrame([
        {'sentence_id': 's1', 'attr_value': 'A'},
        {'sentence_id': 's2', 'attr_value': 'A'},
        {'sentence_id': 's3', 'attr_value': 'B'},
        {'sentence_id': 's4', 'attr_value': 'B'},
    ])
    
    df_ca = perform_correspondence_analysis(df_tokens, df_sentences, attr_col='attr_value', top_k=10)
    
    assert df_ca is not None
    assert isinstance(df_ca, pd.DataFrame)
    # The columns should be name, x, y, type
    assert list(df_ca.columns) == ['name', 'x', 'y', 'type']
    
    # It should have 4 rows (2 words + 2 attributes)
    assert len(df_ca) == 4
    
    # Check that names are correct
    names = df_ca['name'].tolist()
    assert '猫' in names
    assert '犬' in names
    assert 'A' in names
    assert 'B' in names
    
    # Check that types are correct
    types = df_ca['type'].tolist()
    assert types.count('単語') == 2
    assert types.count('属性') == 2

def test_perform_correspondence_analysis_insufficient_data():
    # Less than 2 distinct words
    df_tokens = pd.DataFrame([
        {'word': '猫', 'sentence_id': 's1'},
        {'word': '猫', 'sentence_id': 's2'},
    ])
    df_sentences = pd.DataFrame([
        {'sentence_id': 's1', 'attr_value': 'A'},
        {'sentence_id': 's2', 'attr_value': 'B'},
    ])
    
    df_ca = perform_correspondence_analysis(df_tokens, df_sentences, attr_col='attr_value')
    assert df_ca is None

def test_perform_correspondence_analysis_empty():
    df_tokens = pd.DataFrame(columns=['word', 'sentence_id'])
    df_sentences = pd.DataFrame(columns=['sentence_id', 'attr_value'])
    df_ca = perform_correspondence_analysis(df_tokens, df_sentences, attr_col='attr_value')
    assert df_ca is None

def test_perform_correspondence_analysis_exception():
    # Pass None to trigger exception handling
    df_ca = perform_correspondence_analysis(None, None, None)
    assert df_ca is None


# --- Tests for perform_full_analysis (Mocked) ---

def test_perform_full_analysis_mocked():
    # Prepare mock documents mapped to text sentences
    sentence_map = {
        '吾輩は可愛い猫である': [
            MockToken('吾輩', '吾輩', 'NOUN'),
            MockToken('は', 'は', 'ADP'),
            MockToken('可愛い', '可愛い', 'ADJ'),
            MockToken('猫', '猫', 'NOUN'),
            MockToken('で', 'だ', 'AUX'),
            MockToken('ある', 'ある', 'VERB'),
        ],
        '名前はまだ無い': [
            MockToken('名前', '名前', 'NOUN'),
            MockToken('は', 'は', 'ADP'),
            MockToken('まだ', 'まだ', 'ADV'),
            MockToken('無い', '無い', 'ADJ'),
        ],
        '猫は美しい': [
            MockToken('猫', '猫', 'NOUN'),
            MockToken('は', 'は', 'ADP'),
            MockToken('美しい', '美しい', 'ADJ'),
        ],
        '犬も美しい': [
            MockToken('犬', '犬', 'NOUN'),
            MockToken('も', 'も', 'ADP'),
            MockToken('美しい', '美しい', 'ADJ'),
        ]
    }
    
    mock_nlp = MockNLP(sentence_map)
    mock_sent_dict = {
        '美しい': 'p',
        '可愛い': 'p',
        '無い': 'n',
    }
    
    # Input DataFrame
    df = pd.DataFrame([
        {'text': '吾輩は可愛い猫である。名前はまだ無い。', 'attr': 'A'},
        {'text': '猫は美しい。犬も美しい。', 'attr': 'B'},
    ])
    
    progress_calls = []
    def progress_callback(pct, msg):
        progress_calls.append((pct, msg))
        
    selected_pos = ['NOUN', 'VERB', 'ADJ']
    stop_words = []
    
    with patch('src.core.stats.load_ginza_model', return_value=mock_nlp), \
         patch('src.core.stats.load_sentiment_dict', return_value=mock_sent_dict):
         
         df_freq, df_tfidf, df_ngrams, df_edges, df_sentences, df_tokens, df_ca, corpus_stats = perform_full_analysis(
             df=df,
             text_col='text',
             attr_col='attr',
             selected_pos=selected_pos,
             stop_words=stop_words,
             import_type='CSV / Excel ファイル',
             progress_callback=progress_callback
         )
         
    # 1. Verify progress callback was called and completed at 100%
    assert len(progress_calls) > 0
    assert progress_calls[-1][0] == 100
    
    # 2. Verify corpus_stats
    # Total sentences = 4
    # Raw tokens (ignoring punctuation/spaces):
    # s1: 吾輩(NOUN), は(ADP), 可愛い(ADJ), 猫(NOUN), だ(AUX), ある(VERB) -> 6
    # s2: 名前(NOUN), は(ADP), まだ(ADV), 無い(ADJ) -> 4
    # s3: 猫(NOUN), は(ADP), 美しい(ADJ) -> 3
    # s4: 犬(NOUN), も(ADP), 美しい(ADJ) -> 3
    # Total raw tokens = 6 + 4 + 3 + 3 = 16
    # Unique lemmas: 吾輩, は, 可愛い, 猫, だ, ある, 名前, まだ, 無い, 美しい, 犬, も -> 12
    assert corpus_stats['総文数'] == 4
    assert corpus_stats['総単語数 (前処理前)'] == 16
    assert corpus_stats['異なり単語数 (前処理前)'] == 12
    assert abs(corpus_stats['語彙多様性指数 (TTR)'] - (12/16)) < 1e-5
    assert abs(corpus_stats['平均文長 (単語数)'] - 4.0) < 1e-5
    
    # 3. Verify sentiment classes/scores
    # s1: 可愛い (p) -> pos=1, neg=0 -> score = 1.0 (ポジティブ)
    # s2: 無い (n) -> pos=0, neg=1 -> score = -1.0 (ネガティブ)
    # s3: 美しい (p) -> pos=1, neg=0 -> score = 1.0 (ポジティブ)
    # s4: 美しい (p) -> pos=1, neg=0 -> score = 1.0 (ポジティブ)
    sent_classes = df_sentences['class'].tolist()
    assert sent_classes[0] == 'ポジティブ'
    assert sent_classes[1] == 'ネガティブ'
    assert sent_classes[2] == 'ポジティブ'
    assert sent_classes[3] == 'ポジティブ'
    
    # 4. Verify TF-IDF calculation
    assert not df_tfidf.empty
    assert '平均重要度スコア' in df_tfidf.columns
    # Check that highest TF-IDF words are from our selected pos
    for word in df_tfidf['word']:
        assert word in ['吾輩', '可愛い', '猫', 'ある', '名前', '無い', '美しい', '犬']
        
    # 5. Verify N-grams (Bigrams and Trigrams)
    # s1 words: 吾輩, 可愛い, 猫, ある (all are in selected_pos)
    # Bigrams: 吾輩 - 可愛い, 可愛い - 猫, 猫 - ある
    # Trigrams: 吾輩 - 可愛い - 猫, 可愛い - 猫 - ある
    # s2 words: 名前, 無い (both NOUN/ADJ)
    # Bigrams: 名前 - 無い
    # s3 words: 猫, 美しい
    # Bigrams: 猫 - 美しい
    # s4 words: 犬, 美しい
    # Bigrams: 犬 - 美しい
    ngram_list = df_ngrams['連語'].tolist()
    assert '吾輩 - 可愛い' in ngram_list
    assert '吾輩 - 可愛い - 猫' in ngram_list
    assert '名前 - 無い' in ngram_list
    
    # 6. Verify Jaccard co-occurrence edge calculations
    # Let's check co-occurrences of words:
    # Sentence s1 tokens: 吾輩, 可愛い, 猫, ある
    # Sentence s2 tokens: 名前, 無い
    # Sentence s3 tokens: 猫, 美しい
    # Sentence s4 tokens: 犬, 美しい
    # Wait, '猫' appears in s1 and s3 -> df(猫) = 2
    # '美しい' appears in s3 and s4 -> df(美しい) = 2
    # '猫' and '美しい' co-occur in s3 -> cooc(猫, 美しい) = 1
    # Jaccard = 1 / (2 + 2 - 1) = 1/3 ≈ 0.333333
    row_ne_be = df_edges[((df_edges['word1'] == '猫') & (df_edges['word2'] == '美しい')) |
                         ((df_edges['word1'] == '美しい') & (df_edges['word2'] == '猫'))]
    assert len(row_ne_be) == 1
    jaccard_val = row_ne_be.iloc[0]['jaccard']
    assert abs(jaccard_val - (1/3)) < 1e-4
    
    # '名前' appears in s2 -> df(名前) = 1
    # '無い' appears in s2 -> df(無い) = 1
    # Co-occurrence = 1 -> Jaccard = 1 / (1 + 1 - 1) = 1.0
    row_na_na = df_edges[((df_edges['word1'] == '名前') & (df_edges['word2'] == '無い')) |
                         ((df_edges['word1'] == '無い') & (df_edges['word2'] == '名前'))]
    assert len(row_na_na) == 1
    assert abs(row_na_na.iloc[0]['jaccard'] - 1.0) < 1e-4

    # 7. Verify SVD correspondence analysis was run (df_ca is not None since attr_col is set)
    assert df_ca is not None
    assert isinstance(df_ca, pd.DataFrame)
    
def test_perform_full_analysis_empty_words_raise_error():
    # DataFrame with no words that match POS filter or no words at all
    df = pd.DataFrame([{'text': '。 ！', 'attr': 'A'}])
    with pytest.raises(ValueError, match="解析対象の単語が見つかりませんでした"):
        perform_full_analysis(df, 'text', 'attr', ['NOUN'], [], 'CSV / Excel ファイル')


# --- Integration Test with Real spaCy/ja_ginza model ---

def test_perform_full_analysis_real():
    # Verify that the actual pipeline runs end-to-end with the real spaCy model
    df = pd.DataFrame([
        {'text': '吾輩は猫である。名前はまだ無い。', 'attr': 'A'},
        {'text': '猫は可愛い。犬も可愛い。', 'attr': 'B'},
    ])
    
    selected_pos = ['NOUN', 'PROPN', 'VERB', 'ADJ']
    stop_words = []
    
    df_freq, df_tfidf, df_ngrams, df_edges, df_sentences, df_tokens, df_ca, corpus_stats = perform_full_analysis(
        df=df,
        text_col='text',
        attr_col='attr',
        selected_pos=selected_pos,
        stop_words=stop_words,
        import_type='CSV / Excel ファイル'
    )
    
    # Assert return types and structures
    assert isinstance(df_freq, pd.DataFrame)
    assert isinstance(df_tfidf, pd.DataFrame)
    assert isinstance(df_ngrams, pd.DataFrame)
    assert isinstance(df_edges, pd.DataFrame)
    assert isinstance(df_sentences, pd.DataFrame)
    assert isinstance(df_tokens, pd.DataFrame)
    assert isinstance(df_ca, pd.DataFrame)
    assert isinstance(corpus_stats, dict)
    
    # Verify we got sentences
    assert len(df_sentences) == 4
    # Verify we parsed tokens
    assert len(df_tokens) > 0
    # Verify corpus stats properties
    assert corpus_stats['総文数'] == 4
    assert corpus_stats['総単語数 (前処理前)'] > 0

import os
import pytest
from unittest.mock import patch
import pandas as pd
from src.core.nlp_engine import load_ginza_model, load_sentiment_dict

def test_load_sentiment_dict_success():
    # Test loading actual sentiment dictionary
    sent_dict = load_sentiment_dict()
    assert isinstance(sent_dict, dict)
    # Check that there are entries
    assert len(sent_dict) > 0
    # Let's check polarity of some known words in assets/sentiment_dict.csv
    # e.g., 'あいた 口 が ふさがる ない' has polarity 'n'
    assert 'あいた 口 が ふさがる ない' in sent_dict
    assert sent_dict['あいた 口 が ふさがる ない'] == 'n'

def test_load_sentiment_dict_file_missing():
    # Test load_sentiment_dict when dictionary file does not exist
    load_sentiment_dict.clear()
    with patch('os.path.exists', return_value=False):
        sent_dict = load_sentiment_dict()
        assert sent_dict == {}

def test_load_sentiment_dict_read_csv_exception():
    # Test load_sentiment_dict when pd.read_csv raises an exception
    load_sentiment_dict.clear()
    with patch('os.path.exists', return_value=True), \
         patch('pandas.read_csv', side_effect=Exception("Read error")):
        sent_dict = load_sentiment_dict()
        assert sent_dict == {}

def test_load_ginza_model_real():
    # Verify that the model loads and behaves as expected for basic morphology analysis
    nlp = load_ginza_model()
    doc = nlp("吾輩は猫である")
    
    tokens = [(token.text, token.lemma_, token.pos_) for token in doc]
    assert len(tokens) > 0
    
    # Check token "吾輩"
    assert tokens[0][0] == "吾輩"
    assert tokens[0][1] == "吾輩"
    assert tokens[0][2] in ("NOUN", "PRON")

def test_morphology_analysis_token_properties():
    # Verify morphological properties of spaCy tokens using real model
    nlp = load_ginza_model()
    doc = nlp("！。、 \n")
    for token in doc:
        if token.text in ("！", "。", "、"):
            assert token.is_punct
        elif token.text in (" ", "\n"):
            assert token.is_space

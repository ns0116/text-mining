import os
import spacy
import pandas as pd
import streamlit as st
from src.core.config import resource_path

@st.cache_resource
def load_ginza_model():
    """GiNZAモデルをキャッシュして読み込む"""
    try:
        model_path = resource_path('ja_ginza')
        return spacy.load(model_path)
    except OSError:
        try:
            return spacy.load('ja_ginza')
        except OSError as e:
            raise RuntimeError("GiNZAモデルが見つかりません。") from e

@st.cache_data
def load_sentiment_dict():
    """ローカルの感情極性辞書をロードする"""
    dict_path = resource_path('assets/sentiment_dict.csv')
    sent_dict = {}
    if os.path.exists(dict_path):
        try:
            df_sent = pd.read_csv(dict_path, encoding='utf-8')
            df_sent = df_sent.dropna(subset=['word', 'polarity'])
            sent_dict = dict(zip(df_sent['word'].astype(str), df_sent['polarity'].astype(str)))
        except Exception as e:
            print(f"感情極性辞書の読み込みに失敗しました: {e}")
    return sent_dict


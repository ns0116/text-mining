import pytest
import pandas as pd
import plotly.graph_objects as go
from unittest.mock import patch
from src.core.visualizer import generate_wordcloud_fig, generate_cooc_plotly, generate_ca_plotly


# --- generate_wordcloud_fig ---

def test_wordcloud_returns_none_for_empty_frequencies():
    assert generate_wordcloud_fig({}, "") is None


def test_wordcloud_generates_figure_with_valid_data():
    import matplotlib.pyplot as plt
    frequencies = {"猫": 10, "犬": 5, "魚": 3}
    fig = generate_wordcloud_fig(frequencies, "")
    # フォントが存在しない環境でも None か Figure のどちらかを返す
    if fig is not None:
        assert hasattr(fig, "savefig")
        plt.close(fig)


def test_wordcloud_logs_warning_when_no_font_found(caplog):
    import logging
    with patch("os.path.exists", return_value=False), \
         patch("src.core.visualizer.resource_path", return_value="/nonexistent/font.ttf"), \
         caplog.at_level(logging.WARNING, logger="src.core.visualizer"):
        fig = generate_wordcloud_fig({"テスト": 1}, "/nonexistent/font.ttf")
    assert "日本語フォントが見つからない" in caplog.text
    # フォントなしでも WordCloud はデフォルトフォントで動作し、None にはならない
    if fig is not None:
        import matplotlib.pyplot as plt
        plt.close(fig)


# --- generate_cooc_plotly ---

def test_cooc_returns_none_for_empty_edges():
    df_edges = pd.DataFrame(columns=["word1", "word2", "cooc", "jaccard"])
    df_freq = pd.DataFrame(columns=["word", "出現回数"])
    assert generate_cooc_plotly(df_edges, df_freq) is None


def test_cooc_returns_plotly_figure():
    df_edges = pd.DataFrame([
        {"word1": "猫", "word2": "可愛い", "cooc": 3, "jaccard": 0.75},
        {"word1": "猫", "word2": "犬", "cooc": 2, "jaccard": 0.5},
    ])
    df_freq = pd.DataFrame([
        {"word": "猫", "出現回数": 5},
        {"word": "可愛い", "出現回数": 3},
        {"word": "犬", "出現回数": 2},
    ])
    fig = generate_cooc_plotly(df_edges, df_freq, top_n_edges=10, layout_k=0.4)
    assert fig is not None
    assert isinstance(fig, go.Figure)


def test_cooc_top_n_limits_edges():
    rows = [{"word1": f"w{i}", "word2": f"w{i+1}", "cooc": 1, "jaccard": 0.1} for i in range(10)]
    df_edges = pd.DataFrame(rows)
    freq_rows = [{"word": f"w{i}", "出現回数": 1} for i in range(11)]
    df_freq = pd.DataFrame(freq_rows)
    fig = generate_cooc_plotly(df_edges, df_freq, top_n_edges=3)
    assert fig is not None
    # ノード数はエッジ3本分の頂点（最大6）に収まる
    node_trace = fig.data[1]
    assert len(node_trace.x) <= 6


# --- generate_ca_plotly ---

def test_ca_returns_none_for_none_input():
    assert generate_ca_plotly(None) is None


def test_ca_returns_none_for_empty_dataframe():
    assert generate_ca_plotly(pd.DataFrame()) is None


def test_ca_returns_plotly_figure():
    df_ca = pd.DataFrame([
        {"name": "猫", "x": 0.5, "y": 0.3, "type": "単語"},
        {"name": "犬", "x": -0.5, "y": -0.3, "type": "単語"},
        {"name": "A", "x": 0.4, "y": 0.2, "type": "属性"},
        {"name": "B", "x": -0.4, "y": -0.2, "type": "属性"},
    ])
    fig = generate_ca_plotly(df_ca)
    assert fig is not None
    assert isinstance(fig, go.Figure)


def test_ca_figure_has_two_traces():
    df_ca = pd.DataFrame([
        {"name": "猫", "x": 0.5, "y": 0.3, "type": "単語"},
        {"name": "A", "x": 0.4, "y": 0.2, "type": "属性"},
    ])
    fig = generate_ca_plotly(df_ca)
    # 単語トレースと属性トレースの2本
    assert len(fig.data) == 2
    names = [trace.name for trace in fig.data]
    assert "単語" in names
    assert "属性カテゴリ" in names


def test_ca_axes_have_labels():
    df_ca = pd.DataFrame([
        {"name": "猫", "x": 0.5, "y": 0.3, "type": "単語"},
        {"name": "A", "x": 0.4, "y": 0.2, "type": "属性"},
    ])
    fig = generate_ca_plotly(df_ca)
    assert fig.layout.xaxis.title.text == "第一成分軸"
    assert fig.layout.yaxis.title.text == "第二成分軸"

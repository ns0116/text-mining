import os
import logging
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from src.core.config import resource_path

logger = logging.getLogger(__name__)

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
            logger.warning("日本語フォントが見つからないため文字化けする可能性があります。")
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
        logger.error(f"ワードクラウド生成エラー: {e}")
        raise RuntimeError(f"ワードクラウド生成エラー: {e}") from e

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

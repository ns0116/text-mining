# -*- mode: python ; coding: utf-8 -*-
import os
import streamlit
import ja_ginza
import ginza
import sudachidict_core
import matplotlib
from PyInstaller.utils.hooks import copy_metadata

# Get paths dynamically
streamlit_static = os.path.join(os.path.dirname(streamlit.__file__), 'static')
ja_ginza_model = os.path.join(os.path.dirname(ja_ginza.__file__), 'ja_ginza-5.2.0')
ginza_dir = os.path.dirname(ginza.__file__)
sudachidict_core_dir = os.path.dirname(sudachidict_core.__file__)
matplotlib_mpl_data = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data')

datas = [
    ('assets/NotoSansJP-Regular.ttf', 'assets'), 
    ('src/text_mining_app.py', '.'), 
    ('src/core', 'src/core'),
    (streamlit_static, 'streamlit/static'), 
    (ja_ginza_model, 'ja_ginza'), 
    (ginza_dir, 'ginza'), 
    (sudachidict_core_dir, 'sudachidict_core'), 
    (matplotlib_mpl_data, 'matplotlib/mpl-data')
]
datas += copy_metadata('streamlit')
datas += copy_metadata('spacy')
datas += copy_metadata('spacy-legacy')
datas += copy_metadata('ginza')
datas += copy_metadata('ja-ginza')
datas += copy_metadata('catalogue')
datas += copy_metadata('confection')
datas += copy_metadata('sudachipy')
datas += copy_metadata('sudachidict-core')
datas += copy_metadata('pandas')
datas += copy_metadata('scikit-learn')
datas += copy_metadata('scipy')
datas += copy_metadata('matplotlib')
datas += copy_metadata('plotly')


a = Analysis(
    ['src/run.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['plotly.io._kaleido', 'spacy.lang.ja', 'spacy_legacy.architectures', 'sudachipy', 'sudachipy.morpheme', 'streamlit.runtime.scriptrunner.magic_funcs', 'wordcloud', 'sklearn.utils._cython_blas', 'scipy.special.cython_special', 'sklearn.feature_extraction.text', 'spacy_legacy.architectures.tok2vec', 'spacy_legacy.architectures.entity_linker', 'spacy_legacy.architectures.tagger', 'spacy_legacy.architectures.textcat', 'spacy_legacy.architectures.parser', 'spacy_legacy.architectures.ner', 'spacy_legacy.architectures.attribute_ruler', 'spacy_legacy.architectures.lemmatizer'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='TextMiningApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# text-mining — Claude Context

日本語テキストマイニング用の Streamlit アプリ（形態素解析・可視化）。

## スタック

- Python / Streamlit / GiNZA (ja_ginza) / scikit-learn / matplotlib / altair
- venv: `.venv/`（gitignore対象）
- 現在の本体: `src/text_mining_app.py`（`_attic/` 配下の app_v100.py 等は旧版・参照専用、編集しない）

## 実行

```bash
scripts\text_mining_app_launch.bat
# 中身: .venv を有効化 → streamlit run src\text_mining_app.py
```

## 注意点

- `_attic/` は過去バージョンのアーカイブ。新機能はここに書かず `src/` 側で作業する
- `assets/NotoSansJP-Regular.ttf` は日本語グラフ描画用フォント

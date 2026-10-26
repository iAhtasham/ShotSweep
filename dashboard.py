"""
Dashboard for screenshot_sorter.db.   Run:  python dashboard.py
(or: streamlit run dashboard.py)

Browse images by predicted label and score, mark them 'delete' / 'keep',
and export the marked paths to a text file. Nothing is deleted from here.
"""
import io
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from streamlit import runtime

if not runtime.exists():
    # started as `python dashboard.py` -> relaunch under the streamlit server
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", __file__, *sys.argv[1:]]))

from screenshot_sorter import DB_PATH, LABELS, classify

st.set_page_config(page_title="WhatsApp Screenshot Sorter", layout="wide")


def con():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data(ttl=5)
def load():
    cols = ", ".join(LABELS)
    return pd.read_sql(
        f"SELECT path, folder, size_bytes, width, height, top_label, {cols}, mark "
        "FROM images WHERE top_label IS NOT NULL", con())


@st.cache_data(max_entries=2000)
def thumb(path, size=220):
    try:
        with Image.open(path) as im:
            im.draft("RGB", (size, size))
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((size, size))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
            return buf.getvalue()
    except Exception:
        return None


def set_mark(paths, value):
    c = con()
    c.executemany("UPDATE images SET mark = ? WHERE path = ?", [(value, p) for p in paths])
    c.commit()
    load.clear()


if not Path(DB_PATH).exists():
    st.error(f"{DB_PATH} not found - run `python screenshot_sorter.py embed` and `classify` first.")
    st.stop()

# ---------------------------------------------------------------- classify
st.sidebar.header("Classify")
embedded, unscored = con().execute(
    "SELECT COUNT(emb), COUNT(emb) - COUNT(top_label) FROM images").fetchone()
st.sidebar.caption(f"{embedded:,} images embedded, {unscored:,} not yet classified")
if st.sidebar.button("Run classify", disabled=not embedded, width="stretch"):
    with st.spinner(f"Classifying {embedded:,} images..."):
        n, labels, _ = classify()
    load.clear()
    st.sidebar.success(f"Classified {n:,}: " + ", ".join(f"{k} {v:,}" for k, v in labels.items()))

df = load()
if df.empty:
    st.warning("No classified images yet - run `python screenshot_sorter.py classify`.")
    st.stop()

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Filter")
label = st.sidebar.selectbox("Score by label", list(LABELS))
lo, hi = st.sidebar.slider("Score range", 0.0, 1.0, (0.7, 1.0), 0.01)
folders = st.sidebar.multiselect("Folders", sorted(df.folder.unique()))
mark_f = st.sidebar.radio("Mark", ["any", "unmarked", "delete", "keep"], horizontal=True)
order = st.sidebar.radio("Sort", ["score ↓", "score ↑"], horizontal=True)
per_page = st.sidebar.select_slider("Per page", [24, 48, 96, 192], 48)
cols_n = st.sidebar.slider("Columns", 3, 10, 6)

f = df[(df[label] >= lo) & (df[label] <= hi)]
if folders:
    f = f[f.folder.isin(folders)]
if mark_f == "unmarked":
    f = f[f["mark"].isna()]
elif mark_f != "any":
    f = f[f["mark"] == mark_f]
f = f.sort_values(label, ascending=order == "score ↑")

# ---------------------------------------------------------------- overview
st.title("WhatsApp Screenshot Sorter")
gb = 1024 ** 3
m = st.columns(5)
m[0].metric("Images", f"{len(df):,}")
m[1].metric("Top label = screenshot", f"{(df.top_label == 'screenshot').sum():,}")
m[2].metric("In current filter", f"{len(f):,}", f"{f.size_bytes.sum() / gb:.2f} GB", delta_color="off")
marked = df[df["mark"] == "delete"]
m[3].metric("Marked delete", f"{len(marked):,}", f"{marked.size_bytes.sum() / gb:.2f} GB", delta_color="off")
m[4].metric("Marked keep", f"{(df['mark'] == 'keep').sum():,}")

with st.expander("Charts", expanded=False):
    c1, c2 = st.columns(2)
    c1.caption("Images per top label")
    c1.bar_chart(df.top_label.value_counts())
    c2.caption(f"Distribution of '{label}' score")
    bins = pd.cut(df[label], bins=[i / 20 for i in range(21)], include_lowest=True)
    c2.bar_chart(bins.value_counts().sort_index().rename(lambda iv: f"{iv.left:.2f}"))
    st.caption("Images per folder × top label")
    st.dataframe(pd.crosstab(df.folder, df.top_label), width="stretch")

# ---------------------------------------------------------------- bulk actions
b = st.columns(4)
if b[0].button(f"Mark all {len(f):,} in filter → delete", type="primary"):
    set_mark(f.path, "delete"); st.rerun()
if b[1].button(f"Mark all {len(f):,} in filter → keep"):
    set_mark(f.path, "keep"); st.rerun()
if b[2].button("Clear marks in filter"):
    set_mark(f.path, None); st.rerun()
b[3].download_button(
    f"Export {len(marked):,} 'delete' paths", "\n".join(marked.path),
    file_name="marked_delete.txt", mime="text/plain")

# ---------------------------------------------------------------- grid
pages = max(1, -(-len(f) // per_page))
page = st.number_input(f"Page (of {pages})", 1, pages, 1)
view = f.iloc[(page - 1) * per_page: page * per_page]

grid = st.columns(cols_n)
for i, r in enumerate(view.itertuples(index=False)):
    with grid[i % cols_n]:
        data = thumb(r.path)
        if data:
            st.image(data, width="stretch")
        tag = {"delete": "🗑", "keep": "✅"}.get(r.mark, "")
        st.caption(f"{tag} {getattr(r, label):.2f} · {r.top_label} · {r.width}×{r.height}\n\n{Path(r.path).name}")
        k1, k2 = st.columns(2)
        if k1.button("🗑", key="d" + r.path, help="mark delete"):
            set_mark([r.path], "delete"); st.rerun()
        if k2.button("✅", key="k" + r.path, help="mark keep"):
            set_mark([r.path], "keep"); st.rerun()

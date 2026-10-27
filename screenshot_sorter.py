"""
Offline screenshot finder for WhatsApp image folders.

Uses CLIP (openai/clip-vit-base-patch32) zero-shot classification on CPU.
The model is downloaded once (~600 MB) into the HuggingFace cache; after that
everything runs offline (set HF_HUB_OFFLINE=1 to enforce it).

Everything is stored in a SQLite database (screenshot_sorter.db). Nothing is
moved or deleted.

  1. python screenshot_sorter.py embed      # slow, resumable: CLIP embedding per image
  2. python screenshot_sorter.py classify   # fast: label scores for every image
  3. streamlit run dashboard.py             # browse / mark / export
"""
import argparse
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# WhatsApp media folder; override with the WHATSAPP_DIR environment variable
ROOT = Path(os.environ.get("WHATSAPP_DIR", Path(__file__).with_name("Whatsapp")))
DEFAULT_DIRS = [ROOT / "WhatsApp Images", ROOT / "GBWhatsApp Images"]
DB_PATH = Path(__file__).with_name("screenshot_sorter.db")
MODEL_ID = "openai/clip-vit-base-patch32"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Label -> prompt ensemble.
LABELS = {
    "screenshot": [
        "a screenshot of a phone screen",
        "a mobile phone screenshot",
        "a screenshot of a chat conversation",
        "a screenshot of a WhatsApp chat",
        "a screenshot of a social media post",
        "a screenshot of a website",
        "a screenshot of an app user interface",
        "a screenshot of a computer screen",
        "a screenshot of a text message",
    ],
    "text_graphic": [
        "a meme with text",
        "a good morning greeting image with text",
        "a religious quote image",
        "a poster with text",
        "a digital flyer or advertisement",
    ],
    "document": [
        "a photo of a paper document",
        "a scanned document",
        "a photo of a receipt",
        "a photo of handwritten notes",
    ],
    "photo": [
        "a photo of people",
        "a selfie",
        "a group photo",
        "a photo of a child",
        "a photo taken outdoors",
        "a landscape photo",
        "a photo of food",
        "a photo of an object",
        "a photo of a room",
        "a photo of an animal",
        "a photo of a car",
        "a photo of an event or wedding",
    ],
}

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS images (
    path       TEXT PRIMARY KEY,
    folder     TEXT,
    size_bytes INTEGER,
    mtime      REAL,
    width      INTEGER,   -- -1 = unreadable
    height     INTEGER,
    emb        BLOB,      -- float16[512], L2-normalised CLIP image embedding
    top_label  TEXT,
    {", ".join(f"{k} REAL" for k in LABELS)},
    mark       TEXT       -- set from the dashboard: 'delete' / 'keep' / NULL
);
CREATE INDEX IF NOT EXISTS idx_top ON images(top_label);
CREATE INDEX IF NOT EXISTS idx_folder ON images(folder);
{"".join(f"CREATE INDEX IF NOT EXISTS idx_{k} ON images({k});" for k in LABELS)}
"""


def connect():
    # generous timeout: embed, classify and the dashboard may write concurrently
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.executescript(SCHEMA)
    return con


def list_images(dirs):
    out = []
    for d in dirs:
        for dp, _, files in os.walk(d):
            for f in files:
                if Path(f).suffix.lower() in EXTS:
                    out.append(os.path.join(dp, f))
    out.sort()
    return out


def folder_of(path):
    rel = Path(path).relative_to(ROOT)
    return str(rel.parent)


def load_model():
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    # 8 is as fast as 16 here and leaves cores for the image-decoding threads
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    model = CLIPModel.from_pretrained(MODEL_ID).eval()
    tok = CLIPTokenizer.from_pretrained(MODEL_ID)
    return model, tok


def features(out):
    # transformers 5 returns an output object, older versions a tensor
    return out if hasattr(out, "norm") else out.pooler_output


# ---------------------------------------------------------------- embedding

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
BLANK = np.zeros((3, 224, 224), np.float32)


def preprocess(path, size=224):
    """CLIP preprocessing; returns (array CHW float32, width, height)."""
    from PIL import Image, ImageOps

    try:
        with Image.open(path) as im:
            w, h = im.size
            im.draft("RGB", (size * 2, size * 2))  # fast JPEG downscale on decode
            im = ImageOps.exif_transpose(im).convert("RGB")
            # pad to square instead of center-crop so tall screenshots keep
            # their status bar / UI chrome, which is the strongest cue
            im = ImageOps.pad(im, (size, size), method=Image.BICUBIC, color=(0, 0, 0))
            a = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
            return a.transpose(2, 0, 1), w, h
    except Exception:
        return BLANK, -1, -1


def cmd_embed(args):
    import torch

    con = connect()
    paths = list_images(args.dirs)
    done = {r[0] for r in con.execute("SELECT path FROM images WHERE emb IS NOT NULL OR width = -1")}
    todo = [p for p in paths if p not in done]
    print(f"{len(paths)} images, {len(done)} already embedded, {len(todo)} to go")
    if not todo:
        return

    model, _ = load_model()
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    n = 0
    # threads, not processes: PIL decoding releases the GIL, and worker
    # processes each load their own torch and exhaust memory on Windows
    with ThreadPoolExecutor(args.threads) as pool, torch.inference_mode():
        # keep only a few batches in flight; pool.map would queue all of them
        ahead = 3
        futures = [pool.submit(lambda b=b: [preprocess(p) for p in b]) for b in batches[:ahead]]
        for bi, batch in enumerate(batches):
            prepped = futures[bi].result()
            futures[bi] = None
            if bi + ahead < len(batches):
                b = batches[bi + ahead]
                futures.append(pool.submit(lambda b=b: [preprocess(p) for p in b]))
            x = torch.from_numpy(np.stack([p[0] for p in prepped]))
            e = torch.nn.functional.normalize(features(model.get_image_features(pixel_values=x)), dim=-1)
            e = e.numpy().astype(np.float16)
            rows = []
            for k, (path, (_, w, h)) in enumerate(zip(batch, prepped)):
                st = os.stat(path)
                rows.append((path, folder_of(path), st.st_size, st.st_mtime, w, h,
                             e[k].tobytes() if w > 0 else None))
            con.executemany(
                "INSERT OR REPLACE INTO images (path, folder, size_bytes, mtime, width, height, emb) "
                "VALUES (?,?,?,?,?,?,?)", rows)
            con.commit()
            n += len(batch)
            print(f"\r{n}/{len(todo)}", end="", flush=True)
    print("\ndone")


# ---------------------------------------------------------------- classify

def text_embeddings(model, tok):
    import torch

    out = []
    with torch.inference_mode():
        for prompts in LABELS.values():
            t = tok(prompts, padding=True, return_tensors="pt")
            e = torch.nn.functional.normalize(features(model.get_text_features(**t)), dim=-1).mean(0)
            out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    return np.stack(out)


def classify():
    """Score every embedded image; returns (n, top-label counts, screenshot threshold counts)."""
    con = connect()
    rows = con.execute("SELECT path, emb FROM images WHERE emb IS NOT NULL").fetchall()
    if not rows:
        return 0, {}, {}
    emb = np.stack([np.frombuffer(r[1], np.float16) for r in rows]).astype(np.float32)

    model, tok = load_model()
    logits = 100.0 * emb @ text_embeddings(model, tok).T  # CLIP logit scale
    logits -= logits.max(1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(1, keepdims=True)

    names = list(LABELS)
    top = prob.argmax(1)
    con.executemany(
        f"UPDATE images SET top_label = ?, {', '.join(f'{k} = ?' for k in names)} WHERE path = ?",
        [(names[top[i]], *map(float, prob[i]), r[0]) for i, r in enumerate(rows)])
    con.commit()

    s = prob[:, names.index("screenshot")]
    return (
        len(rows),
        {name: int((top == k).sum()) for k, name in enumerate(names)},
        {t: int((s >= t).sum()) for t in (0.5, 0.6, 0.7, 0.8, 0.9)},
    )


def cmd_classify(args):
    n, labels, thresholds = classify()
    if not n:
        sys.exit("no embeddings yet - run `embed` first")
    print(f"classified {n} images in {DB_PATH}")
    for name, c in labels.items():
        print(f"  {name:13s} top-label: {c:6d}")
    for t, c in thresholds.items():
        print(f"  screenshot prob >= {t}: {c}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("embed")
    p.add_argument("--dirs", nargs="+", type=Path, default=DEFAULT_DIRS)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--threads", type=int, default=4, help="image decoding threads")
    p.set_defaults(fn=cmd_embed)

    sub.add_parser("classify").set_defaults(fn=cmd_classify)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

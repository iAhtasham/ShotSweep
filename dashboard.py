"""
Dashboard for screenshot_sorter.db.   Run:  python dashboard.py

Browse images by predicted label and score, view them full size, mark them
'delete' / 'keep', and export the marked paths. Nothing is deleted from here.
Serves on 127.0.0.1 only, so your images are never exposed to the network.
"""
import argparse
import io
import sqlite3
import threading
import webbrowser
from functools import lru_cache
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file
from PIL import Image, ImageOps

from screenshot_sorter import DB_PATH, LABELS, classify

app = Flask(__name__)
PAGE = Path(__file__).with_name("dashboard.html")

# height / width ranges for the shape filter
SHAPES = {
    "wide": (0, 0.7),
    "landscape": (0.7, 0.9),
    "square": (0.9, 1.1),
    "portrait": (1.1, 1.5),
    "tall": (1.5, 1.9),
    "phone": (1.9, 99),
}
MARKS = {"delete", "keep"}
job = {"running": False, "message": ""}


def db():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def where(args):
    """Build the WHERE clause shared by listing, bulk marking and counting."""
    label = args.get("label", "screenshot")
    if label not in LABELS:
        abort(400)
    sql = [f"top_label IS NOT NULL", f"{label} >= ?", f"{label} <= ?"]
    params = [float(args.get("lo", 0)), float(args.get("hi", 1))]
    if folder := args.get("folder"):
        sql.append("folder = ?"); params.append(folder)
    mark = args.get("mark", "any")
    if mark == "unmarked":
        sql.append("mark IS NULL")
    elif mark in MARKS:
        sql.append("mark = ?"); params.append(mark)
    if (shape := args.get("shape")) in SHAPES:
        lo, hi = SHAPES[shape]
        sql.append("height * 1.0 / width >= ? AND height * 1.0 / width < ?"); params += [lo, hi]
    return label, " AND ".join(sql), params


@app.get("/")
def index():
    return send_file(PAGE)


@app.get("/api/summary")
def summary():
    con = db()
    r = con.execute("""
        SELECT COUNT(*) total,
               SUM(width = -1) empty,
               COUNT(emb) embedded,
               COUNT(top_label) classified,
               SUM(mark = 'delete') del_n,
               SUM(CASE WHEN mark = 'delete' THEN size_bytes END) del_bytes,
               SUM(mark = 'keep') keep_n
        FROM images""").fetchone()
    tops = dict(con.execute(
        "SELECT top_label, COUNT(*) FROM images WHERE top_label IS NOT NULL GROUP BY top_label").fetchall())
    folders = [f for (f,) in con.execute(
        "SELECT DISTINCT folder FROM images WHERE top_label IS NOT NULL ORDER BY folder")]
    return jsonify({**dict(r), "labels": list(LABELS), "tops": tops, "folders": folders, "job": job})


@app.get("/api/histogram")
def histogram():
    label = request.args.get("label", "screenshot")
    if label not in LABELS:
        abort(400)
    rows = db().execute(
        f"SELECT MIN(CAST({label} * 20 AS INT), 19) b, COUNT(*) FROM images "
        f"WHERE top_label IS NOT NULL GROUP BY b").fetchall()
    bins = [0] * 20
    for b, n in rows:
        bins[b] = n
    return jsonify(bins)


@app.get("/api/images")
def images():
    label, sql, params = where(request.args)
    order = "ASC" if request.args.get("sort") == "asc" else "DESC"
    offset, limit = int(request.args.get("offset", 0)), min(int(request.args.get("limit", 120)), 500)
    con = db()
    total, size = con.execute(f"SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM images WHERE {sql}", params).fetchone()
    rows = con.execute(
        f"SELECT rowid id, path, folder, width w, height h, size_bytes size, top_label top, mark, {label} score "
        f"FROM images WHERE {sql} ORDER BY {label} {order}, rowid LIMIT ? OFFSET ?",
        params + [limit, offset]).fetchall()
    items = [{**dict(r), "name": Path(r["path"]).name} for r in rows]
    for it in items:
        del it["path"]
    return jsonify(total=total, bytes=size, items=items)


@app.post("/api/mark")
def mark():
    body = request.get_json()
    value = body.get("mark")
    if value not in MARKS | {None}:
        abort(400)
    con = db()
    if "ids" in body:
        con.executemany("UPDATE images SET mark = ? WHERE rowid = ?", [(value, int(i)) for i in body["ids"]])
        n = len(body["ids"])
    else:  # everything matching the filter
        _, sql, params = where(body.get("filter", {}))
        n = con.execute(f"UPDATE images SET mark = ? WHERE {sql}", [value] + params).rowcount
    con.commit()
    return jsonify(updated=n)


def path_of(image_id):
    r = db().execute("SELECT path FROM images WHERE rowid = ?", (image_id,)).fetchone()
    if not r or not Path(r["path"]).is_file():
        abort(404)
    return r["path"]


@lru_cache(maxsize=4000)
def thumbnail(path, size=360):
    with Image.open(path) as im:
        im.draft("RGB", (size, size))  # decode JPEGs at reduced size: much faster
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((size, size), Image.BICUBIC)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=82)
        return buf.getvalue()


@app.get("/thumb/<int:image_id>")
def thumb(image_id):
    try:
        data = thumbnail(path_of(image_id))
    except OSError:
        abort(415)
    resp = Response(data, mimetype="image/jpeg")
    resp.cache_control.max_age = 86400
    return resp


@app.get("/image/<int:image_id>")
def full_image(image_id):
    return send_file(path_of(image_id), max_age=86400)


@app.get("/export")
def export():
    paths = [p for (p,) in db().execute("SELECT path FROM images WHERE mark = 'delete' ORDER BY path")]
    return Response("\n".join(paths) + "\n", mimetype="text/plain",
                    headers={"Content-Disposition": "attachment; filename=marked_delete.txt"})


@app.post("/api/classify")
def run_classify():
    if job["running"]:
        return jsonify(job)

    def work():
        try:
            n, tops, _ = classify()
            job["message"] = f"Classified {n:,} images"
        except Exception as e:
            job["message"] = f"Classify failed: {e}"
        job["running"] = False

    job.update(running=True, message="Classifying… (loading model)")
    threading.Thread(target=work, daemon=True).start()
    return jsonify(job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if not DB_PATH.exists():
        raise SystemExit(f"{DB_PATH} not found - run `python screenshot_sorter.py embed` first")
    url = f"http://127.0.0.1:{args.port}"
    print(f"Dashboard: {url}")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()

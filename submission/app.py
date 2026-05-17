"""
Video Scene Search — Gradio web interface.

Improvements over the starter template:

* **Click-to-seek** — clicking a result thumbnail jumps the video player
  to that timestamp.
* **Batch query mode** — paste a newline-separated list of queries and
  download one JSON file with all results.
* **JSON + CSV export** of the most recent search.
* **Live processing stats** + a per-query latency display.
* **Results table** alongside the thumbnail gallery.

Run with::

    python -m submission.app
    # or
    cd submission && python app.py
"""

from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

try:
    from main import VideoSearch  # when launched from inside submission/
except ImportError:  # pragma: no cover
    from submission.main import VideoSearch


_searcher: Optional[VideoSearch] = None
_current_video: Optional[str] = None
_last_results: List = []

EXPORT_DIR = Path("exports").absolute()
EXPORT_DIR.mkdir(exist_ok=True)


def get_searcher() -> VideoSearch:
    global _searcher
    if _searcher is None:
        _searcher = VideoSearch()
    return _searcher


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
def _format_stats_md(stats: dict) -> str:
    if not stats:
        return "_No video loaded yet._"
    mi = stats.get("model_info", {}) or {}
    mods = ", ".join(stats.get("modalities", []) or ["clip-visual"])
    return (
        f"**Frames** {stats.get('sampled_frames', 0)} sampled / "
        f"{stats.get('total_frames', 0)} total &nbsp;·&nbsp; "
        f"**{stats.get('scenes_detected', 0)}** scenes detected  \n"
        f"**Index time** {stats.get('index_time_seconds', 0):.2f}s &nbsp;·&nbsp; "
        f"**FPS** {stats.get('fps', 0):.1f} &nbsp;·&nbsp; "
        f"**RAM** {stats.get('memory_mb', 0):.0f} MB &nbsp;·&nbsp; "
        f"**Cache** {'hit' if stats.get('cache_hit') else 'fresh'}  \n"
        f"**Modalities** {mods} &nbsp;·&nbsp; "
        f"**Model** {mi.get('version', 'n/a')} on `{mi.get('device', 'cpu')}` "
        f"(dim={mi.get('embedding_dim', '?')})"
    )


def load_video(video_file) -> Tuple[str, str, Optional[str]]:
    global _current_video, _last_results
    _last_results = []
    if video_file is None:
        return "Please upload a video file.", _format_stats_md({}), None
    video_path = video_file if isinstance(video_file, str) else getattr(
        video_file, "name", str(video_file),
    )
    try:
        searcher = get_searcher()
        t0 = time.time()
        searcher.load_video(video_path)
        elapsed = time.time() - t0
        _current_video = video_path
        stats = searcher.get_processing_stats()
        msg = (
            f"Loaded **{Path(video_path).name}** in **{elapsed:.1f}s** "
            f"({'cache hit' if stats.get('cache_hit') else 'fresh index'}). "
            f"Ready to search."
        )
        return msg, _format_stats_md(stats), video_path
    except Exception as e:
        return f"Error loading video: {e}", _format_stats_md({}), video_path


def _conf_dot(c: float) -> str:
    """Color-coded confidence indicator."""
    if c >= 0.66:
        return f"🟢 {c:.3f}"
    if c >= 0.33:
        return f"🟡 {c:.3f}"
    return f"🔴 {c:.3f}"


def _results_to_rows(results) -> List[List]:
    return [
        [i, f"{r.timestamp_ms / 1000:.2f}s", r.frame_number,
         _conf_dot(r.confidence), r.thumbnail_path or ""]
        for i, r in enumerate(results, 1)
    ]


def _results_to_gallery(results):
    out = []
    for i, r in enumerate(results, 1):
        cap = f"#{i}  {r.timestamp_ms / 1000:.2f}s  conf {r.confidence:.2f}"
        out.append((r.thumbnail_path, cap) if r.thumbnail_path else (None, cap))
    return out


def search_video(query: str, top_k: int):
    global _last_results
    if not query or not query.strip():
        return [], [], "_Please enter a search query._"
    if _current_video is None:
        return [], [], "_Please load a video first._"
    try:
        searcher = get_searcher()
        t0 = time.time()
        results = searcher.search(query.strip(), top_k=int(top_k))
        latency_ms = (time.time() - t0) * 1000
        _last_results = results
        gallery = _results_to_gallery(results)
        rows = _results_to_rows(results)
        if not results:
            msg = f"No results for '_{query}_' · **{latency_ms:.1f} ms**"
        else:
            top = results[0]
            msg = (
                f"**{len(results)}** results for '_{query}_' · "
                f"**{latency_ms:.1f} ms** · "
                f"top hit at **{top.timestamp_ms / 1000:.2f}s** "
                f"(conf {top.confidence:.2f}) · "
                f"_click a thumbnail to seek_"
            )
        return gallery, rows, msg
    except Exception as e:
        return [], [], f"Error: {e}"


def batch_search(queries_text: str, top_k: int):
    if _current_video is None:
        return "Please load a video first.", None
    lines = [q.strip() for q in (queries_text or "").splitlines() if q.strip()]
    if not lines:
        return "Paste one query per line.", None
    searcher = get_searcher()
    out = {}
    for q in lines:
        try:
            res = searcher.search(q, top_k=int(top_k))
            out[q] = [
                {"timestamp_ms": r.timestamp_ms,
                 "frame_number": r.frame_number,
                 "confidence": round(r.confidence, 4),
                 "thumbnail_path": r.thumbnail_path}
                for r in res
            ]
        except Exception as e:
            out[q] = {"error": str(e)}
    path = EXPORT_DIR / "batch_results.json"
    path.write_text(json.dumps(out, indent=2))
    return (
        f"Ran **{len(lines)}** queries · wrote `{path.name}` "
        f"({sum(len(v) if isinstance(v, list) else 0 for v in out.values())} total results)."
    ), str(path)


def export_json() -> Optional[str]:
    if not _last_results:
        return None
    data = [
        {"timestamp_ms": r.timestamp_ms, "frame_number": r.frame_number,
         "confidence": r.confidence, "thumbnail_path": r.thumbnail_path}
        for r in _last_results
    ]
    path = EXPORT_DIR / "last_results.json"
    path.write_text(json.dumps(data, indent=2))
    return str(path)


def export_csv() -> Optional[str]:
    if not _last_results:
        return None
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["rank", "timestamp_ms", "frame_number", "confidence", "thumbnail_path"])
    for i, r in enumerate(_last_results, 1):
        w.writerow([i, r.timestamp_ms, r.frame_number,
                    f"{r.confidence:.4f}", r.thumbnail_path or ""])
    path = EXPORT_DIR / "last_results.csv"
    path.write_text(buf.getvalue())
    return str(path)


def seek_to_result(evt: gr.SelectData):
    if not _last_results or evt.index is None or _current_video is None:
        return gr.update(), gr.update()
    idx = evt.index if isinstance(evt.index, int) else evt.index[0]
    if idx >= len(_last_results):
        return gr.update(), gr.update()
    r = _last_results[idx]
    seconds = r.timestamp_ms / 1000.0
    note = (
        f"▶ Seeking to **#{idx + 1}** at **{seconds:.2f}s** "
        f"(frame {r.frame_number}, conf {r.confidence:.2f})"
    )
    return (_current_video, seconds), note


def clear_results():
    global _last_results
    _last_results = []
    return [], [], "_Cleared._", ""


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
/* =========================================================================
   Soothing slate-blue palette — calming, professional, AAA-contrast text.
   Page: soft cool off-white. Surfaces: pure white. Text: near-black slate.
   Primary: muted slate-blue. Accents: powder blue / soft sage.
   ========================================================================= */

:root {
    --bg-page:        #0f1419;   /* deep near-black */
    --bg-card:        #1a212b;   /* dark slate surface */
    --bg-muted:       #232b37;   /* slightly lighter surface */
    --bg-accent:      #2a3344;   /* hover / accent surface */
    --border:         #2d3748;
    --border-strong:  #475569;
    --text:           #f1f5f9;   /* near-white body */
    --text-strong:    #ffffff;   /* headings */
    --text-muted:     #cbd5e1;
    --primary:        #6889a8;   /* slate-blue, bright on dark */
    --primary-hover:  #88a8c8;
    --accent-sage:    #84a98c;
    --accent-warn:    #d4a574;
    --accent-err:     #e57b6f;
}

.gradio-container {
    max-width: 1200px !important;
    background: var(--bg-page) !important;
    color: var(--text) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
.gradio-container * { color: var(--text); }
.gradio-container p, .gradio-container span, .gradio-container li,
.gradio-container td, .gradio-container th, .gradio-container div { color: var(--text); }

/* ---------- Hero ---------- */
#hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #2c4f73 55%, #3d6691 100%);
    padding: 26px 28px;
    border-radius: 16px;
    border: 1px solid #3d6691;
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.55);
    margin-bottom: 18px;
}
#hero, #hero * { color: #ffffff !important; }
#hero h1 { margin: 0 0 8px 0; font-size: 1.75em; font-weight: 700; letter-spacing: -0.015em; }
#hero p  { margin: 0; color: #cfe0f2 !important; line-height: 1.55; }
#hero b  { color: #ffffff !important; font-weight: 600; }

/* ---------- Cards & surfaces ---------- */
.card, #stats-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
}
#stats-card { border-left: 4px solid var(--primary) !important; }

.gradio-container .block,
.gradio-container .form,
.gradio-container .wrap,
.gradio-container .gr-box,
.gradio-container .file-preview,
.gradio-container .upload-container,
.gradio-container .gallery,
.gradio-container .table-wrap,
.gradio-container .gr-panel {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    border-radius: 10px;
}

/* ---------- Inputs ---------- */
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container textarea,
.gradio-container select,
.gradio-container .gr-input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(104, 137, 168, 0.30) !important;
    outline: none !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
    font-weight: 500 !important;
}
/* Upload area hints ("- or -") and file-name secondary text */
.gradio-container .or, .gradio-container [class*="or"] {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}

/* ---------- Labels ---------- */
.gradio-container label,
.gradio-container .label,
.gradio-container label span,
.gradio-container .label-wrap span,
.gradio-container .block-label,
.gradio-container [class*="block-label"],
.gradio-container [class*="label-wrap"] > span:first-child {
    background: transparent !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* ---------- Tables ---------- */
.gradio-container table, .gradio-container th, .gradio-container td {
    background: var(--bg-card) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
.gradio-container thead th, .gradio-container .table-wrap thead {
    background: var(--bg-muted) !important;
    color: var(--text) !important;
    font-weight: 700 !important;
    border-bottom: 2px solid var(--border-strong) !important;
}
.gradio-container tbody tr:hover td { background: var(--bg-accent) !important; }

/* ---------- Now-playing & search-message pills ----------
   Only show pill styling when there is actual markdown content.
   When empty, collapse entirely so users don't see blank coloured bars. */
#now-playing, #search-message {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    color: var(--text) !important;
    font-size: 0.92rem;
    min-height: 0 !important;
}
#now-playing:has(p), #now-playing:has(.prose p) {
    background: #1f3328 !important;
    border-left: 4px solid var(--accent-sage) !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
}
#search-message:has(p), #search-message:has(.prose p) {
    background: #1e2c3f !important;
    border-left: 4px solid var(--primary) !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
}
/* Strong, dark text inside the pills */
#now-playing *, #search-message * { color: var(--text) !important; }

/* ---------- Gallery ---------- */
.gallery .thumbnail-item, .grid-wrap .thumbnail-item {
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    overflow: hidden;
}
.gallery .thumbnail-item:hover, .grid-wrap .thumbnail-item:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 22px rgba(0, 0, 0, 0.55) !important;
    border-color: var(--primary) !important;
}

/* ---------- Buttons ---------- */
button.primary, .primary > button {
    background: var(--primary) !important;
    color: #ffffff !important;
    border: 1px solid var(--primary-hover) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: background 0.15s ease, transform 0.05s ease;
}
button.primary:hover, .primary > button:hover {
    background: var(--primary-hover) !important;
}
button.primary:active, .primary > button:active { transform: translateY(1px); }

button.secondary, .secondary > button,
.gradio-container button:not(.primary):not(.icon-button) {
    background: var(--bg-card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-strong) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
button.secondary:hover, .secondary > button:hover,
.gradio-container button:not(.primary):not(.icon-button):hover {
    background: var(--bg-muted) !important;
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}

/* ---------- Example chips ---------- */
.gradio-container .examples button,
.gradio-container [class*="example"] button {
    background: var(--bg-muted) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    padding: 6px 14px !important;
    font-weight: 500 !important;
}
.gradio-container .examples button:hover {
    background: var(--bg-accent) !important;
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}

/* ---------- Links ---------- */
.gradio-container a, .gradio-container a:visited {
    color: var(--primary) !important;
    text-decoration: underline;
    text-underline-offset: 2px;
}
.gradio-container a:hover { color: var(--primary-hover) !important; }

/* ---------- Markdown / code ---------- */
.gradio-container .prose, .gradio-container .markdown { color: var(--text) !important; }
.gradio-container .prose code, .gradio-container code {
    background: var(--bg-muted) !important;
    color: var(--primary-hover) !important;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.88em;
}

/* ---------- Accordion ---------- */
.gradio-container .accordion, .gradio-container [class*="accordion"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ---------- Slider ---------- */
.gradio-container input[type="range"] {
    accent-color: var(--primary) !important;
}

/* ---------- Video player — theater-style dark frame ---------- */
#video-player {
    background: linear-gradient(160deg, #0f172a 0%, #1a2332 100%) !important;
    border: 1px solid #1a2332 !important;
    border-radius: 14px !important;
    padding: 8px !important;
    box-shadow: 0 8px 28px rgba(15, 23, 42, 0.22), 0 0 0 1px rgba(255,255,255,0.04) inset !important;
    overflow: hidden !important;
}
#video-player .label-wrap, #video-player [class*="label-wrap"] {
    background: transparent !important;
}
#video-player .label-wrap span, #video-player [class*="label-wrap"] span {
    color: #e7eef6 !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em;
}
#video-player video,
#video-player [data-testid="video"],
#video-player [class*="video-container"] {
    border-radius: 10px !important;
    background: #000000 !important;
    overflow: hidden !important;
}
#video-player [class*="empty"], #video-player [class*="Empty"] {
    color: #94a3b8 !important;
    background: transparent !important;
}
#video-player [class*="empty"] svg {
    color: #64748b !important;
    opacity: 0.8;
}
/* Bottom control bar inside the player */
#video-player [class*="controls"], #video-player [class*="Controls"] {
    background: linear-gradient(180deg, transparent 0%, rgba(15,23,42,0.85) 100%) !important;
    border-radius: 0 0 10px 10px !important;
    padding: 8px 10px !important;
}
#video-player [class*="time"], #video-player time {
    color: #e7eef6 !important;
    font-variant-numeric: tabular-nums;
    font-weight: 500;
}
#video-player progress, #video-player [role="progressbar"] {
    accent-color: #6889a8 !important;
    height: 4px !important;
    border-radius: 2px !important;
    background: rgba(255,255,255,0.18) !important;
}
#video-player a[href*="download"], #video-player [class*="download"] button {
    background: rgba(255,255,255,0.10) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 8px !important;
}
#video-player a[href*="download"]:hover, #video-player [class*="download"] button:hover {
    background: rgba(255,255,255,0.18) !important;
    border-color: #ffffff !important;
}

/* ---------- Video player — dark pill controls so play icon is visible ---------- */
.gradio-container [class*="player"] button,
.gradio-container [class*="Player"] button,
.gradio-container [class*="control"] button,
.gradio-container button.play-pause-button,
.gradio-container button[aria-label*="play" i],
.gradio-container button[aria-label*="volume" i],
.gradio-container button[aria-label*="full" i],
.gradio-container button[title*="play" i],
.gradio-container button[title*="pause" i] {
    background: rgba(26, 35, 50, 0.78) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.28) !important;
    border-radius: 999px !important;
}
.gradio-container [class*="player"] button:hover,
.gradio-container button.play-pause-button:hover,
.gradio-container button[aria-label*="play" i]:hover {
    background: var(--text) !important;
    border-color: #ffffff !important;
}
.gradio-container [class*="player"] button img,
.gradio-container [class*="Player"] button img,
.gradio-container [class*="control"] button img,
.gradio-container button.play-pause-button img,
.gradio-container button[aria-label*="play" i] img,
.gradio-container button[aria-label*="volume" i] img,
.gradio-container button[aria-label*="full" i] img {
    filter: brightness(0) invert(1) !important;
}

/* ---------- Footer ---------- */
.gradio-container footer { color: var(--text-muted) !important; }
.gradio-container footer a { color: var(--primary) !important; }
"""


def create_interface() -> Tuple[gr.Blocks, gr.themes.Base]:
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_md,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0f1419",
        body_background_fill_dark="#0f1419",
        body_text_color="#f1f5f9",
        body_text_color_dark="#f1f5f9",
        background_fill_primary="#1a212b",
        background_fill_primary_dark="#1a212b",
        background_fill_secondary="#232b37",
        background_fill_secondary_dark="#232b37",
        block_background_fill="#1a212b",
        block_background_fill_dark="#1a212b",
        block_border_color="#2d3748",
        block_border_color_dark="#2d3748",
        input_background_fill="#1a212b",
        input_background_fill_dark="#1a212b",
        input_border_color="#475569",
        input_border_color_dark="#475569",
        border_color_primary="#2d3748",
        border_color_primary_dark="#2d3748",
        color_accent_soft="#2a3344",
        color_accent_soft_dark="#2a3344",
    )
    with gr.Blocks(title="Multimodal Video Search") as app:
        gr.HTML(
            "<div id='hero'>"
            "<h1>🎬 Multimodal Video Search</h1>"
            "<p>Natural-language search inside any video — "
            "<b>CLIP visual</b> + <b>Whisper speech</b> + <b>OCR text</b>, "
            "fused with <b>Reciprocal Rank Fusion</b>. "
            "Calibrated 0–1 confidence, click any result to seek the player.</p>"
            "</div>"
        )

        with gr.Row(equal_height=False):
            # ---- Left: load + stats ----
            with gr.Column(scale=4):
                video_input = gr.File(
                    label="1. Upload a video",
                    file_types=["video"],
                    type="filepath",
                )
                load_btn = gr.Button("🚀 Load & Index", variant="primary", size="lg")
                load_status = gr.Markdown("_No video loaded yet._")
                stats_output = gr.Markdown(_format_stats_md({}), elem_id="stats-card")

            # ---- Right: player + query ----
            with gr.Column(scale=6):
                video_player = gr.Video(
                    label="Preview (auto-seeks when you click a result)",
                    interactive=False,
                    height=380,
                    elem_id="video-player",
                )
                now_playing = gr.Markdown("", elem_id="now-playing")
                query_input = gr.Textbox(
                    label="2. Search query",
                    placeholder="e.g. a person wearing a red dress",
                    lines=1,
                    autofocus=True,
                )
                with gr.Row():
                    top_k_slider = gr.Slider(1, 30, value=10, step=1,
                                             label="Top-K", scale=3)
                    search_btn = gr.Button("🔍 Search", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        search_message = gr.Markdown("", elem_id="search-message")
        results_gallery = gr.Gallery(
            label="3. Results — click any thumbnail to seek the video",
            columns=5, rows=2, height=320, object_fit="cover",
            allow_preview=True, show_label=True,
        )
        results_table = gr.Dataframe(
            headers=["#", "time", "frame", "confidence", "thumbnail"],
            datatype=["number", "str", "number", "str", "str"],
            interactive=False, wrap=True, row_count=(0, "dynamic"),
        )

        with gr.Accordion("📦 Batch query mode — run many queries at once", open=False):
            gr.Markdown(
                "Paste **one query per line**. Each query runs through the same "
                "search pipeline and the combined results are written to a "
                "single JSON file (timestamps, frame numbers, confidences, "
                "thumbnail paths) — useful for evaluation, regression checks, "
                "or feeding into a downstream tool."
            )
            batch_text = gr.Textbox(
                lines=6, label="Queries (one per line)",
                placeholder=(
                    "a person walking\n"
                    "two people talking\n"
                    "car driving\n"
                    "close-up of a face"
                ),
            )
            batch_btn = gr.Button("Run batch", variant="secondary")
            batch_status = gr.Markdown("")
            batch_file = gr.File(label="Download batch JSON")

        with gr.Accordion("Export last search", open=False):
            with gr.Row():
                json_btn = gr.Button("Export JSON")
                csv_btn = gr.Button("Export CSV")
            export_json_file = gr.File(label="JSON export", visible=False)
            export_csv_file = gr.File(label="CSV export", visible=False)

        gr.Markdown("### Example queries (click to fill the box, then press Enter)")
        gr.Examples(
            examples=[
                ["a person walking"],
                ["two people talking to each other"],
                ["a crowded or busy scene"],
                ["someone holding an object"],
                ["outdoor scene with trees or nature"],
            ],
            inputs=query_input,
        )

        # ---- Wiring ----
        load_btn.click(load_video, [video_input],
                       [load_status, stats_output, video_player])
        search_btn.click(search_video, [query_input, top_k_slider],
                         [results_gallery, results_table, search_message])
        query_input.submit(search_video, [query_input, top_k_slider],
                           [results_gallery, results_table, search_message])
        results_gallery.select(seek_to_result,
                               outputs=[video_player, now_playing])
        json_btn.click(export_json, outputs=[export_json_file]).then(
            lambda p: gr.update(visible=p is not None, value=p),
            inputs=[export_json_file], outputs=[export_json_file],
        )
        csv_btn.click(export_csv, outputs=[export_csv_file]).then(
            lambda p: gr.update(visible=p is not None, value=p),
            inputs=[export_csv_file], outputs=[export_csv_file],
        )
        batch_btn.click(batch_search, [batch_text, top_k_slider],
                        [batch_status, batch_file])
        clear_btn.click(clear_results, outputs=[results_gallery, results_table,
                                                search_message, now_playing])

    return app, theme


app, THEME = create_interface()

if __name__ == "__main__":
    # allow Gradio to serve thumbnails from the on-disk cache directory
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=THEME,
        css=CUSTOM_CSS,
        allowed_paths=[str(VideoSearch.THUMB_DIR), str(EXPORT_DIR)],
    )

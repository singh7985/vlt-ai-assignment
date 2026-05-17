---
title: Multimodal Video Search
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.0.0"
app_file: app.py
pinned: true
license: mit
short_description: Find any scene in a video using natural language — CLIP + Whisper + OCR fused with RRF.
---

# 🎬 Multimodal Video Search

> **Find any scene in a video by describing it in plain English.**
> Built by **Aman Singh** as an AI/ML Engineer interview submission for **vlt-ai**.

Combines three modalities — **CLIP** (visual), **faster-whisper** (speech), and **RapidOCR** (on-screen text) — fused with **Reciprocal Rank Fusion** for calibrated 0–1 confidence scores.

---

## ▶️ Try it live

| Where | Link |
|---|---|
| **Live demo** (no install) | https://huggingface.co/spaces/`<your-hf-username>`/multimodal-video-search |
| **Source code** | https://github.com/`<your-gh-username>`/multimodal-video-search |
| **Walkthrough** | [`submission/README.md`](submission/README.md) |
| **Eval framework** | [`EVALUATION.md`](EVALUATION.md) |
| **Original brief** | [`ASSIGNMENT.md`](ASSIGNMENT.md) |

> Recruiter shortcut: open the live demo → upload any short MP4 (or click an example query) → click a result thumbnail to seek the player.

---

## ⚡ Run locally in 3 commands

```bash
git clone https://github.com/<your-gh-username>/multimodal-video-search.git
cd multimodal-video-search
pip install -r requirements.txt && python app.py     # → http://localhost:7860
```

`ffmpeg` is required for audio extraction:
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt install -y ffmpeg`

Open the URL and:
1. **Upload** any video (1–10 min works best).
2. Click **🚀 Load & Index** (≈ 30–60 s on first run, cached after).
3. Type a query like *"a person wearing a red dress"* or *"someone says thank you"*.
4. Click any result thumbnail — the player **auto-seeks** to that moment.

---

## 🧠 How it works

```
        ┌──────────────┐
video → │ scene detect │ → keyframes ─┐
        └──────────────┘              │
                                      ▼
        ┌──────────────┐         ┌──────────┐         ┌─────┐
audio → │   Whisper    │ → text ▶│ CLIP-text│         │     │
        └──────────────┘         │  encoder │ ──────▶ │ RRF │ ──▶ ranked results
        ┌──────────────┐         └──────────┘         │     │
frames→ │  RapidOCR    │ → text ─────────▲            │     │
        └──────────────┘                 │            └─────┘
                                  query ─┘
```

- **CLIP ViT-B/32** scores each keyframe against the text query.
- **Whisper** transcripts and **OCR** captions are scored by lexical overlap.
- **Reciprocal Rank Fusion** merges all three rankings — robust to score scale differences.
- Confidences are **calibrated to [0, 1]** so the UI's 🟢🟡🔴 dots actually mean something.

---

## 🧪 Tests & evaluation

```bash
pytest submission/tests -q                       # 9/9 unit tests
python -m evaluation.evaluate --check-interface --submission ./submission
python -m evaluation.evaluate --submission ./submission \
    --video ./data/your_video.mp4 \
    --queries ./data/sample_queries.json \
    --output report.json
```

---

## 📂 Repo layout

```
.
├── app.py                  # Hugging Face Spaces entry point
├── requirements.txt        # full deployment deps
├── submission/             # actual project — VideoSearch + Gradio UI
│   ├── main.py             #   VideoSearchInterface implementation
│   ├── app.py              #   Gradio interface (dark theme, click-to-seek)
│   ├── search/             #   CLIP, Whisper, OCR, RRF modules
│   ├── tests/              #   pytest suite
│   └── README.md           #   deep-dive write-up
├── evaluation/             # grading harness (unchanged from brief)
├── ASSIGNMENT.md           # original interview brief
└── EVALUATION.md           # scoring criteria
```

---

## 📜 License

MIT — see [`LICENSE`](LICENSE) if present, otherwise this file constitutes the license grant.

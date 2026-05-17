# Deployment Guide

This project ships to two destinations:

| Target | Purpose |
|---|---|
| **GitHub** | Source, history, CI badge. |
| **Hugging Face Space** | Hosted Gradio demo (free CPU tier). |

The root [`app.py`](../app.py), [`requirements.txt`](../requirements.txt), [`packages.txt`](../packages.txt), and the HF frontmatter in [`README.md`](../README.md) are already configured for both targets.

---

## 1. GitHub

```bash
# from the repo root
git remote rename origin upstream   # keep the template repo around as 'upstream'
git remote add origin https://github.com/singh7985/democlip.git
git push -u origin master
```

Authenticate with a Personal Access Token (Settings → Developer settings → Tokens → Fine-grained → repo write).

Suggested repo metadata:

- **Description:** *Multimodal video search — CLIP + Whisper + OCR fused with RRF.*
- **Topics:** `gradio`, `clip`, `whisper`, `video-search`, `multimodal`, `ocr`

---

## 2. Hugging Face Space

1. Create the Space at <https://huggingface.co/new-space>
   - **SDK:** Gradio · **Gradio version:** `6.0.0` · **Hardware:** CPU basic (free) · **License:** MIT
2. Push:
   ```bash
   git remote add hf https://huggingface.co/spaces/singh7985/democlip
   git push hf master:main
   ```
   Password = an HF access token with **write** scope (<https://huggingface.co/settings/tokens>).
3. First build downloads torch + CLIP + Whisper weights (≈ 4–6 minutes). When status is **Running** the UI appears on the **App** tab.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| HF build fails on `torch` install | Pin `torch==2.2.0` in `requirements.txt` |
| HF Space stuck on "Building" >15 min | **Restart this Space** in the ⋮ menu |
| `git push hf` rejected for large file | Confirm `.gitignore` excludes `*.mp4` and `.venv/` (already configured) |
| "App not running" message | Free Spaces sleep after 48h idle — first request wakes them in ~60 s |
| Dark mode doesn't apply | Append `?__theme=dark` to the URL (also the in-app default) |

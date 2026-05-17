# 🚀 Deploy in 10 minutes — recruiter-ready

Two destinations:

| What | Why | URL pattern |
|---|---|---|
| **Hugging Face Space** | One-click live demo. Recruiter clicks, uploads, searches. No install. | `https://huggingface.co/spaces/<hf-user>/multimodal-video-search` |
| **GitHub repo** | Code review, commit history, README on the front page. | `https://github.com/<gh-user>/multimodal-video-search` |

Put both links at the top of your résumé / LinkedIn / email reply.

---

## Part A — GitHub (5 minutes)

### A1. Create the empty repo
1. Open https://github.com/new
2. Repository name: **`multimodal-video-search`**
3. Visibility: **Public** (recruiters can't see private repos without an invite)
4. Do **NOT** initialise with README, .gitignore, or license — we already have those
5. Click **Create repository**

### A2. Push your code
Copy your GitHub username, then run (replace `<gh-user>`):

```bash
cd /Users/xe/Aman_Singh_VoltAi
git remote rename origin upstream                # keep vlt-ai's repo as 'upstream'
git remote add origin https://github.com/<gh-user>/multimodal-video-search.git
git add -A
git commit -m "Deploy: dark-mode UI, HF Spaces entry point, recruiter README"
git branch -M main
git push -u origin main
```

If asked for a password, paste a **Personal Access Token** (Settings → Developer settings → Tokens → Fine-grained → repo permissions).

### A3. Polish the GitHub repo page
1. Go to `https://github.com/<gh-user>/multimodal-video-search`
2. Click the ⚙️ icon next to **About** (top right)
3. Description: *"Multimodal video search — CLIP + Whisper + OCR fused with RRF. Live demo on Hugging Face Spaces."*
4. Website: paste the HF Space URL (after Part B)
5. Topics: `gradio`, `clip`, `whisper`, `video-search`, `multimodal`, `ocr`, `python`

---

## Part B — Hugging Face Spaces (5 minutes)

### B1. Create a free Hugging Face account
1. Sign up at https://huggingface.co/join (free, instant)
2. Verify your email

### B2. Create the Space
1. Open https://huggingface.co/new-space
2. **Owner:** your username
3. **Space name:** `multimodal-video-search`
4. **License:** MIT
5. **SDK:** Gradio
6. **Gradio version:** `6.0.0`
7. **Hardware:** **CPU basic — Free** (good enough for the demo; upgrade later if needed)
8. **Visibility:** Public
9. Click **Create Space**

### B3. Push your code to the Space
Hugging Face will show a page with a `git clone` URL. Use it like this (replace `<hf-user>`):

```bash
cd /Users/xe/Aman_Singh_VoltAi
git remote add hf https://huggingface.co/spaces/<hf-user>/multimodal-video-search
git push hf main:main
```

When prompted:
- **Username:** your HF username
- **Password:** an **HF access token** with **write** scope — create one at https://huggingface.co/settings/tokens

### B4. Watch it build
1. Go to `https://huggingface.co/spaces/<hf-user>/multimodal-video-search`
2. Click the **Logs** tab — you'll see `pip install` running (≈ 4–6 minutes the first time, downloading torch + CLIP + Whisper weights)
3. When status changes to **Running**, the UI appears on the **App** tab
4. First search after a fresh build takes ~60 s (model warm-up). Subsequent ones are fast.

### B5. Update the README links
Edit `README.md` and replace the two placeholders:
- `<your-hf-username>` → your HF username
- `<your-gh-username>` → your GitHub username

Then:

```bash
git add README.md
git commit -m "docs: add live demo + repo links"
git push origin main
git push hf main:main
```

---

## Part C — Share it (1 minute)

Send recruiters **one line**:

> Live demo: https://huggingface.co/spaces/`<hf-user>`/multimodal-video-search
> Code: https://github.com/`<gh-user>`/multimodal-video-search

Optional polish:
- Add both links to your **résumé header** under your email/LinkedIn.
- Add to **LinkedIn → Featured** (pin the GitHub repo as a featured link).
- Reply to the vlt-ai assignment email with both URLs in the first paragraph.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| HF build fails on `torch` install | Switch hardware to **CPU upgrade** (still free for some accounts) or pin `torch==2.2.0` in `requirements.txt` |
| HF Space stuck on "Building" >15 min | Click **Restart this Space** in the ⋮ menu |
| `git push hf` rejected (large file) | Make sure `.gitignore` excludes `*.mp4` and `.venv/` — already configured |
| Recruiter sees "App not running" | Free Spaces sleep after 48h idle — they tap the screen and it wakes in ~60 s |
| Dark mode looks wrong on HF | Append `?__theme=dark` to the URL — also already the in-app default |

---

## What's already done for you ✅

- ✅ Root [`app.py`](app.py) — HF Spaces entry point (imports `submission.app`)
- ✅ Root [`requirements.txt`](requirements.txt) — full deploy dependencies
- ✅ Root [`packages.txt`](packages.txt) — installs `ffmpeg` on the Space
- ✅ Root [`README.md`](README.md) — has HF Spaces frontmatter + recruiter shortcuts
- ✅ [`.gitignore`](.gitignore) — excludes videos, venv, cache
- ✅ Dark mode hard-coded in the Gradio theme — no system-preference flicker

You only have to do the **clicks and pushes** above.

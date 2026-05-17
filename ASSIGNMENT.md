# AI/ML Engineer Interview Assignment

## Multimodal Video Search System

Build a web application that finds specific scenes in videos using natural language queries.

**Example:** Given a video and the query "person wearing a red dress", your system should return timestamps and thumbnails of all scenes containing someone in a red dress.

---

## The Challenge

Create a video search system that:
1. Accepts a video file
2. Takes natural language queries from users
3. Returns relevant video timestamps with confidence scores
4. Displays results through a web interface

### What We're Evaluating

| Criteria | Weight | Description |
|----------|--------|-------------|
| Search Accuracy | 50% | Precision@K and Recall scores |
| Performance | 30% | Latency, throughput, memory efficiency |
| Code Quality | 15% | Architecture, documentation, testing |
| Innovation | 5% | Creative approaches, UX improvements |

---

## Requirements

### Must Have
- [ ] Implement `VideoSearchInterface` in `submission/main.py`
- [ ] Provide a working web interface (Gradio template provided)
- [ ] Handle videos up to 10 minutes in length
- [ ] Return results within reasonable time (<30s for first result)

### Should Have
- [ ] Support common video formats (MP4, MOV, AVI)
- [ ] Return confidence scores with results
- [ ] Display thumbnails for matched scenes
- [ ] Handle edge cases gracefully

### Nice to Have
- [ ] Real-time search as video loads
- [ ] Batch query support
- [ ] Export results to file
- [ ] Custom UI beyond the template

---

## Getting Started

### 1. Set Up Your Private Repository

To keep your solution private from other candidates:

```bash
# Clone this repository
git clone https://github.com/vlt-ai/ai-interview-assignment.git
cd ai-interview-assignment

# Create a new PRIVATE repository on your GitHub account
# (do NOT fork — forks of public repos are public)
gh repo create ai-interview --private

# Change the remote to point to your private repo
git remote set-url origin https://github.com/<your-github-username>/ai-interview.git

# Push the assignment to your private repo
git push -u origin master
```

> **Important:** Do not fork this repository. Forks of public repos are always public, meaning other candidates could see your work. Use the clone-and-push method above instead.

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install evaluation dependencies
pip install -r requirements.txt

# Install your submission dependencies
pip install -r submission/requirements.txt
```

### 2. Get Test Videos

Download any video (1–10 minutes) from the [Internet Archive](https://archive.org/) — it has thousands of free, public domain videos. See [data/README.md](data/README.md) for suggestions.

```bash
# Example: download a video from archive.org
wget "https://archive.org/download/<video-id>/<filename>.mp4" -O data/test_video.mp4
```

### 3. Implement Your Solution

Edit `submission/main.py` to implement the `VideoSearch` class:

```python
from evaluation.interface import VideoSearchInterface, SearchResult

class VideoSearch(VideoSearchInterface):
    def load_video(self, video_path: str) -> None:
        # Load and preprocess the video
        pass

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        # Find scenes matching the query
        pass

    def get_processing_stats(self) -> dict:
        # Return performance statistics
        pass
```

### 4. Test Your Solution

```bash
# Check interface compliance
python -m evaluation.evaluate --check-interface

# Run evaluation
python -m evaluation.evaluate \
    --submission ./submission \
    --video ./data/test_video.mp4 \
    --queries ./data/sample_queries.json \
    --ground-truth ./data/ground_truth/sample_labels.json

# Launch web interface
python -m submission.app
```

---

## Submission Structure

```
submission/
├── main.py           # Your VideoSearch implementation (required)
├── app.py            # Web interface (required, can modify)
├── requirements.txt  # Your dependencies (required)
└── README.md         # Your documentation (optional but recommended)
```

You may add additional files and modules as needed.

---

## Evaluation

### Automated Testing

You can run the full evaluation locally:

```bash
python -m evaluation.evaluate \
    --submission ./submission \
    --video ./data/test_video.mp4 \
    --queries ./data/sample_queries.json \
    --ground-truth ./data/ground_truth/sample_labels.json
```

### Metrics

| Metric | Weight | Excellent | Good | Needs Work |
|--------|--------|-----------|------|------------|
| Precision@10 | 25% | >0.8 | 0.5-0.8 | <0.5 |
| Recall | 25% | >0.8 | 0.5-0.8 | <0.5 |
| Time to First Result | 15% | <2s | 2-10s | >10s |
| Throughput | 15% | >15 FPS | 5-15 FPS | <5 FPS |
| Memory | 10% | <1GB | 1-2GB | >2GB |
| Robustness | 10% | Handles all cases | Minor issues | Crashes |

See [EVALUATION.md](EVALUATION.md) for detailed scoring rubric.

---

## Rules

1. **Work independently** - This is an individual assessment
2. **Use any resources** - Documentation, papers, existing libraries welcome
3. **Document your approach** - Explain your model choices and trade-offs
4. **No pre-trained end-to-end solutions** - You must implement the search logic
5. **Keep it runnable** - Evaluators must be able to run your code

---

## Timeline

This is designed as a weekend project (1-2 days of focused work).

**Suggested approach:**
- Day 1: Set up environment, choose model approach, implement core search
- Day 2: Optimize performance, handle edge cases, polish UI

---

## Resources

- [API Contract Documentation](docs/API_CONTRACT.md)
- [Getting Started Guide](docs/GETTING_STARTED.md)
- [Evaluation Details](EVALUATION.md)

---

## Submitting Your Work

When you're finished:

1. Ensure all tests pass locally
2. Push your final code to your **private** repository
3. Add `vlt-ai` as a collaborator so we can review your work:
   ```bash
   gh repo edit --add-collaborator vlt-ai
   ```
   Or: repo **Settings** → **Collaborators** → Add **vlt-ai**
4. Send your repo link to your recruiter

---

## Questions?

If you encounter issues:
1. Check the documentation first
2. Review the FAQ in [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
3. Contact your recruiter for clarification

Good luck!

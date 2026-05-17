"""Hugging Face Spaces entry point — launches the submission Gradio app."""
from submission.app import app, THEME, CUSTOM_CSS
from submission.main import VideoSearch
from submission.app import EXPORT_DIR

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=THEME,
        css=CUSTOM_CSS,
        allowed_paths=[str(VideoSearch.THUMB_DIR), str(EXPORT_DIR)],
    )

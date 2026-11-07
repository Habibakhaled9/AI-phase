import sys
import subprocess
from pathlib import Path


def main() -> None:
    """Run the Graph RAG Streamlit application."""
    project_root = Path(__file__).resolve().parent.parent.parent
    app_path = project_root / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(cmd)

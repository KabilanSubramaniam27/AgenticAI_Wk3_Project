from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli


def run() -> None:
    """Launch the SeniorCare Streamlit UI through the installed console command."""
    app_path = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path)]
    raise SystemExit(streamlit_cli.main())

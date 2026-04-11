from __future__ import annotations

from pathlib import Path

import streamlit as st


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
CONSOLE_CSS_PATH = ASSETS_DIR / "console.css"


def inject_theme(mode: str = "desktop") -> None:
    css = CONSOLE_CSS_PATH.read_text(encoding="utf-8")
    if mode == "mobile":
        css += """
        .stApp {
          background:
            radial-gradient(circle at top left, rgba(16, 185, 129, 0.08), transparent 28%),
            #f4f7f5;
          color: #132218;
        }
        .stApp [data-testid="block-container"] {
          max-width: 880px;
          padding-top: 1rem;
        }
        """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def pill_html(label: str, status: str) -> str:
    safe_status = status.replace(" ", "_").lower()
    return f'<span class="console-pill pill-{safe_status}">{label}</span>'

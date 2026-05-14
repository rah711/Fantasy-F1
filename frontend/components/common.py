from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st


def role_badge(role: str) -> None:
    if role == "owner":
        st.success("Access role: OWNER (full access)")
    elif role == "visitor":
        st.info("Access role: VISITOR (read-only)")


def readonly_notice() -> None:
    st.warning("Visitor mode: editing and PR writeback are disabled.")


def render_health_table(health: dict) -> None:
    rows = []
    for name, fh in health.items():
        mtime = None
        if fh.mtime:
            mtime = dt.datetime.fromtimestamp(fh.mtime).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(
            {
                "artifact": name,
                "exists": fh.exists,
                "last_updated": mtime,
                "size_bytes": fh.size_bytes,
                "path": fh.path,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def show_diff(diff_text: str) -> None:
    if not diff_text.strip():
        st.info("No config changes detected.")
    else:
        st.code(diff_text, language="diff")


def file_download_button(path: str | Path, label: str) -> None:
    p = Path(path)
    if not p.exists():
        st.caption(f"Missing: {p}")
        return
    data = p.read_bytes()
    st.download_button(label=label, data=data, file_name=p.name, mime="application/octet-stream")

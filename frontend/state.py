from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_services import load_config_file


def _secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


def owner_password() -> str:
    return _secret("OWNER_PASSWORD", "")


def init_session_state() -> None:
    st.session_state.setdefault("auth_role", None)
    st.session_state.setdefault("draft_cfg", None)
    st.session_state.setdefault("last_recommendation", None)
    st.session_state.setdefault("last_transfer_recommendation", None)


def _ensure_cfg_loaded() -> None:
    if st.session_state.get("draft_cfg") is None:
        st.session_state["draft_cfg"] = load_config_file()


def get_working_config() -> dict[str, Any]:
    _ensure_cfg_loaded()
    return st.session_state["draft_cfg"]


def set_working_config(cfg: dict[str, Any]) -> None:
    st.session_state["draft_cfg"] = cfg


def reset_working_config() -> None:
    st.session_state["draft_cfg"] = load_config_file()


def auth_role() -> str | None:
    return st.session_state.get("auth_role")


def is_owner() -> bool:
    return auth_role() == "owner"


def require_auth() -> None:
    init_session_state()
    # Public-by-default app: drop all users into visitor mode.
    if auth_role() not in {"owner", "visitor"}:
        st.session_state["auth_role"] = "visitor"


def owner_access_controls() -> None:
    if is_owner():
        if st.button("Switch to visitor view", key="owner_switch_visitor", use_container_width=True):
            st.session_state["auth_role"] = "visitor"
            st.session_state["last_recommendation"] = None
            st.session_state["last_transfer_recommendation"] = None
            st.rerun()
        return

    if not owner_password():
        return

    st.text_input(
        "Owner password",
        type="password",
        key="owner_login_password",
        label_visibility="collapsed",
        placeholder="Owner password",
    )
    if st.button("Owner login", key="owner_login_btn", use_container_width=True):
        if st.session_state.get("owner_login_password", "").strip() == owner_password():
            st.session_state["auth_role"] = "owner"
            st.session_state["owner_login_password"] = ""
            st.rerun()
        else:
            st.session_state["owner_login_password"] = ""
            st.error("Invalid owner password")


def github_settings_from_secrets() -> dict[str, str]:
    return {
        "token": _secret("GITHUB_TOKEN", ""),
        "owner": _secret("GITHUB_OWNER", ""),
        "repo": _secret("GITHUB_REPO", ""),
        "base_branch": _secret("GITHUB_BASE_BRANCH", "main"),
    }


def secret_status() -> dict[str, bool]:
    return {
        "OWNER_PASSWORD": bool(owner_password()),
        "GITHUB_TOKEN": bool(_secret("GITHUB_TOKEN", "")),
        "GITHUB_OWNER": bool(_secret("GITHUB_OWNER", "")),
        "GITHUB_REPO": bool(_secret("GITHUB_REPO", "")),
    }

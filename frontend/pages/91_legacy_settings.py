from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components import readonly_notice, role_badge
from frontend.state import auth_role, github_settings_from_secrets, is_owner, logout_button, require_auth, secret_status
from src.ui_services import propose_config_change_via_pr

st.set_page_config(page_title="Settings & Access", layout="wide")
require_auth()

role_badge(auth_role() or "")
logout_button()
if not is_owner():
    readonly_notice()

st.title("Settings & Access")

st.subheader("Access Configuration")
status = secret_status()
for k, v in status.items():
    st.write(f"- {k}: {'configured' if v else 'missing'}")

st.subheader("GitHub Writeback Settings")
settings = github_settings_from_secrets()
st.write(
    {
        "owner": settings.get("owner", ""),
        "repo": settings.get("repo", ""),
        "base_branch": settings.get("base_branch", "main"),
        "token_configured": bool(settings.get("token", "")),
    }
)

st.subheader("Dry-Run PR Plumbing Test")
if st.button("Run dry-run PR check"):
    res = propose_config_change_via_pr(
        updated_cfg_yaml="season:\n  year: 2026\n",
        title="Dry-run config writeback check",
        body="Dry run from settings page",
        settings=settings,
        dry_run=True,
    )
    if res.ok:
        st.success(res.message)
        st.caption(f"Proposed branch name: {res.branch_name}")
    else:
        st.error(res.message)

st.info(
    "Required Streamlit secrets: OWNER_PASSWORD, VISITOR_PASSWORD, GITHUB_TOKEN, GITHUB_OWNER, GITHUB_REPO, GITHUB_BASE_BRANCH."
)

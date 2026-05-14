from __future__ import annotations

from src.ui_services.github_writeback import build_branch_name, propose_config_change_via_pr


def test_build_branch_name_prefix() -> None:
    name = build_branch_name("owner-update")
    assert name.startswith("owner-update-")


def test_propose_config_change_missing_settings() -> None:
    res = propose_config_change_via_pr(
        updated_cfg_yaml="season:\n  year: 2026\n",
        title="test",
        body="test",
        settings={"token": "", "owner": "", "repo": ""},
        dry_run=True,
    )
    assert not res.ok
    assert "Missing GitHub settings" in res.message

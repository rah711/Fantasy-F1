from __future__ import annotations

import base64
import datetime as dt
import os
import re
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class PullRequestResult:
    ok: bool
    branch_name: str
    pr_url: str | None
    message: str


class GitHubWritebackClient:
    def __init__(
        self,
        token: str,
        owner: str,
        repo: str,
        base_branch: str = "main",
        session: requests.Session | None = None,
    ) -> None:
        self.owner = owner
        self.repo = repo
        self.base_branch = base_branch
        self.s = session or requests.Session()
        self.s.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    @property
    def api_base(self) -> str:
        return f"https://api.github.com/repos/{self.owner}/{self.repo}"

    def _req(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        r = self.s.request(method, url, timeout=30, **kwargs)
        if r.status_code >= 400:
            msg = ""
            try:
                msg = r.json().get("message", "")
            except Exception:
                msg = r.text[:300]
            raise RuntimeError(f"GitHub API {method} {url} failed ({r.status_code}): {msg}")
        return r

    def get_ref_sha(self, branch: str) -> str:
        url = f"{self.api_base}/git/ref/heads/{branch}"
        return self._req("GET", url).json()["object"]["sha"]

    def create_branch(self, new_branch: str, from_branch: str | None = None) -> None:
        source_branch = from_branch or self.base_branch
        source_sha = self.get_ref_sha(source_branch)
        url = f"{self.api_base}/git/refs"
        payload = {"ref": f"refs/heads/{new_branch}", "sha": source_sha}
        self._req("POST", url, json=payload)

    def get_file_sha(self, path: str, branch: str) -> str | None:
        url = f"{self.api_base}/contents/{path}?ref={branch}"
        r = self.s.get(url, timeout=30)
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            try:
                msg = r.json().get("message", "")
            except Exception:
                msg = r.text[:300]
            raise RuntimeError(f"GitHub API GET {url} failed ({r.status_code}): {msg}")
        return r.json().get("sha")

    def put_file(self, path: str, content_text: str, commit_message: str, branch: str) -> None:
        sha = self.get_file_sha(path, branch)
        url = f"{self.api_base}/contents/{path}"
        payload: dict[str, Any] = {
            "message": commit_message,
            "content": base64.b64encode(content_text.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha
        self._req("PUT", url, json=payload)

    def create_pull_request(self, title: str, body: str, head_branch: str, base_branch: str | None = None) -> str:
        base = base_branch or self.base_branch
        url = f"{self.api_base}/pulls"
        payload = {"title": title, "body": body, "head": head_branch, "base": base}
        return self._req("POST", url, json=payload).json().get("html_url", "")


def _slugify_branch_part(raw: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw.strip()).strip("-").lower()
    return slug or "update"


def build_branch_name(prefix: str = "ui-config-update") -> str:
    ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{_slugify_branch_part(prefix)}-{ts}"


def _resolve_github_settings(overrides: dict[str, str] | None = None) -> dict[str, str]:
    out = {
        "token": "",
        "owner": "",
        "repo": "",
        "base_branch": "main",
    }
    if overrides:
        out.update({k: v for k, v in overrides.items() if v})

    out["token"] = out["token"] or os.getenv("GITHUB_TOKEN", "")
    out["owner"] = out["owner"] or os.getenv("GITHUB_OWNER", "")
    out["repo"] = out["repo"] or os.getenv("GITHUB_REPO", "")
    out["base_branch"] = out["base_branch"] or os.getenv("GITHUB_BASE_BRANCH", "main")
    return out


def propose_config_change_via_pr(
    updated_cfg_yaml: str,
    title: str,
    body: str,
    config_path: str = "config.yaml",
    branch_prefix: str = "ui-config-update",
    settings: dict[str, str] | None = None,
    dry_run: bool = False,
) -> PullRequestResult:
    gh = _resolve_github_settings(settings)
    missing = [k for k in ("token", "owner", "repo") if not gh.get(k)]
    if missing:
        return PullRequestResult(
            ok=False,
            branch_name="",
            pr_url=None,
            message=f"Missing GitHub settings: {', '.join(missing)}",
        )

    branch_name = build_branch_name(prefix=branch_prefix)
    if dry_run:
        return PullRequestResult(
            ok=True,
            branch_name=branch_name,
            pr_url=None,
            message="Dry run successful (no network writes performed)",
        )

    client = GitHubWritebackClient(
        token=gh["token"],
        owner=gh["owner"],
        repo=gh["repo"],
        base_branch=gh.get("base_branch", "main"),
    )

    try:
        client.create_branch(new_branch=branch_name)
        client.put_file(
            path=config_path,
            content_text=updated_cfg_yaml,
            commit_message=f"chore(config): {title}",
            branch=branch_name,
        )
        pr_url = client.create_pull_request(
            title=title,
            body=body,
            head_branch=branch_name,
            base_branch=gh.get("base_branch", "main"),
        )
        return PullRequestResult(ok=True, branch_name=branch_name, pr_url=pr_url, message="PR created")
    except Exception as exc:
        return PullRequestResult(ok=False, branch_name=branch_name, pr_url=None, message=str(exc))


def propose_files_pr(
    files: dict[str, str],
    title: str,
    body: str,
    branch_prefix: str = "ui-data-update",
    settings: dict[str, str] | None = None,
    dry_run: bool = False,
) -> PullRequestResult:
    """Commit a set of repo-relative paths → text content on a new branch and open a PR.

    Use for any file the UI writes that should persist back to the repo (history.csv,
    competitors.csv, race-result CSVs, etc.).
    """
    gh = _resolve_github_settings(settings)
    missing = [k for k in ("token", "owner", "repo") if not gh.get(k)]
    if missing:
        return PullRequestResult(
            ok=False, branch_name="", pr_url=None,
            message=f"Missing GitHub settings: {', '.join(missing)}",
        )
    if not files:
        return PullRequestResult(ok=False, branch_name="", pr_url=None, message="No files to commit")

    branch_name = build_branch_name(prefix=branch_prefix)
    if dry_run:
        return PullRequestResult(
            ok=True, branch_name=branch_name, pr_url=None,
            message=f"Dry run: would PR {len(files)} file(s) on branch {branch_name}",
        )

    client = GitHubWritebackClient(
        token=gh["token"], owner=gh["owner"], repo=gh["repo"],
        base_branch=gh.get("base_branch", "main"),
    )
    try:
        client.create_branch(new_branch=branch_name)
        for path, content in files.items():
            client.put_file(
                path=path, content_text=content,
                commit_message=f"data: {title} — {path}", branch=branch_name,
            )
        pr_url = client.create_pull_request(
            title=title, body=body, head_branch=branch_name,
            base_branch=gh.get("base_branch", "main"),
        )
        return PullRequestResult(ok=True, branch_name=branch_name, pr_url=pr_url, message="PR created")
    except Exception as exc:
        return PullRequestResult(ok=False, branch_name=branch_name, pr_url=None, message=str(exc))

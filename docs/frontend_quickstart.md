# Frontend Quickstart (Streamlit)

## Run locally

```bash
cd "/Users/sarah-macbookair-midnight/Claude Code/Fantasy F1 Data Sci Model"
PYTHONPATH=. streamlit run frontend/app.py
```

## Configure secrets locally

Create `.streamlit/secrets.toml`:

```toml
OWNER_PASSWORD = "your-owner-password"

GITHUB_TOKEN = "ghp_..."
GITHUB_OWNER = "your-github-user-or-org"
GITHUB_REPO = "Fantasy-F1"
GITHUB_BASE_BRANCH = "main"
```

## Access modes

- **Default (no password)**: visitor read-only mode on public pages
- **Owner password**: unlocks owner workflow pages + edit/PR actions from the sidebar button

## Streamlit Community Cloud

Set the same keys in the app's **Secrets** panel.

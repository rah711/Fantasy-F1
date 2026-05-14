# Frontend Quickstart (Streamlit)

## Run locally

```bash
cd "/Users/sarah-macbookair-midnight/Claude Code/Fantasy F1 Data Sci Model"
PYTHONPATH=. streamlit run frontend/app.py
```

## Configure passwords/secrets locally

Create `.streamlit/secrets.toml`:

```toml
OWNER_PASSWORD = "your-owner-password"
VISITOR_PASSWORD = "your-visitor-password"

GITHUB_TOKEN = "ghp_..."
GITHUB_OWNER = "your-github-user-or-org"
GITHUB_REPO = "Fantasy-F1"
GITHUB_BASE_BRANCH = "main"
```

## Access modes

- **Owner password**: full access (edit draft config, create branch + PR)
- **Visitor password**: read-only access (can view and run safe reads; no edits/PR)

## Streamlit Community Cloud

Set the same keys in the app's **Secrets** panel.

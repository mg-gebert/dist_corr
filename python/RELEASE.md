# Python Release Guide

This project publishes the Python package from the `python/` directory.

## One-time setup

1. In PyPI, create the project `dist-corr` (or claim if already created).
2. In GitHub repo settings:
   - Add environment `pypi`.
   - Optionally require approvals before deployment.
3. In PyPI project settings, add a Trusted Publisher for this repository/workflow:
   - Owner: `<your-github-org-or-user>`
   - Repository: `dist_corr`
   - Workflow: `.github/workflows/python-publish.yml`
   - Environment: `pypi`

## Release steps

1. Bump versions consistently:
   - `python/pyproject.toml` `[project].version`
   - `python/pyproject.toml` `[tool.poetry].version`
   - `python/Cargo.toml` `[package].version`
2. Commit and push to your default branch.
3. Create and push a tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

The publish workflow triggers on tags matching `v*`, builds wheels on Linux/macOS/Windows plus an sdist, then publishes to PyPI using OIDC Trusted Publishing.

## Local verification (optional)

From `python/`:

```bash
poetry install
poetry run pytest
poetry run maturin build --release --sdist --out dist
poetry run python -m pip install twine
poetry run twine check dist/*
```

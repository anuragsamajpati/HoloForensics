# Contributing to HoloForensics

Thanks for your interest in contributing! This document explains how to get set up, propose changes, and follow our workflow.

## Getting Started

- Python: 3.10+
- Install dependencies: `pip install -r requirements.txt`
- Optional system deps: `ffmpeg`
- Django dev server: `python manage.py runserver` from `holoforensics/holoforensics_web/`

## Branching Model

- Create feature branches from `main`: `git checkout -b feature/<short-name>`
- Keep PRs small and focused. One logical change per PR.
- Include a clear description and screenshots for UI changes.

## Code Style

- Python: follow PEP8; run `flake8` locally if possible.
- JS/CSS: keep code modular; avoid inline scripts where feasible.
- Backend: document endpoints in docstrings; return helpful error messages.

## Tests

- If adding complex features, include unit tests (e.g., `pytest`) or a manual test plan.

## Commit Messages

- Use imperative mood and keep them concise, e.g., "Add timeline report to scene report page".

## Pull Requests

- Fill in the PR template.
- Link issues with `Fixes #<id>` when applicable.
- Ensure CI passes before requesting review.

## Large Files

- Avoid committing raw videos or heavy models directly. Use Git LFS or external storage.

## Security

- Never commit secrets or API keys. Use environment variables/.env (ignored).

Thanks for helping improve HoloForensics! 🎉

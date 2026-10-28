# Code Formatting in novasplat

This repository uses Ruff for both linting and formatting.

## Source of Truth

Formatting and linting behavior is defined in:

- `ruff.toml`
- `.pre-commit-config.yaml`
- `.vscode/settings.json`

### Current policy

- Line length: 88
- Python target: 3.12
- Excluded from Ruff checks:
  - `thirdparty/bop_toolkit`
  - `thirdparty/cnos25`
  - `thirdparty/colmap`
  - `thirdparty/dinov3`
  - `thirdparty/gedi`
  - `thirdparty/gsplat`
  - `thirdparty/openmvs`
  - `thirdparty/sam3`
  - `thirdparty/vcg`

## What runs automatically

### In VS Code

For Python files, the workspace is configured to:

- Use Ruff as the default formatter
- Format on save
- Keep import organization available as an explicit save action

This is configured in `.vscode/settings.json`.

Note: with `source.organizeImports: "explicit"`, imports are not always
auto-organized on every save. This setting keeps import organization
intentional/opt-in.

### On git commit

A pre-commit hook runs:

1. `ruff --fix --exit-non-zero-on-fix`
2. `ruff-format`

Important behavior:

- If Ruff auto-fixes files, the commit fails once (by design).
- Re-stage and commit again:
  - `git add -A`
  - `git commit -m "..."`

## New Developer Setup (VS Code)

## 1) Use a Python environment for this repo

Use whichever environment workflow your team prefers.

Option A (local `.venv` in repo root):

```bash
python -m venv .venv
source .venv/bin/activate
```

Option B (virtualenvwrapper):

```bash
workon novasplat
```

In VS Code, make sure the selected Python interpreter is the one for this repo:

- Command Palette -> `Python: Select Interpreter`
- Choose your project environment (for example `.venv`)

## 2) Install Python tooling in the environment

```bash
pip install -r requirements.txt
```

This installs Ruff and pre-commit versions pinned for the repo.

## 3) Install VS Code extensions

Install these extensions:

- Python (`ms-python.python`)
- Ruff (`charliermarsh.ruff`)

## 4) Ensure workspace settings are used

Open the repo folder in VS Code so `.vscode/settings.json` is applied.

## 5) Install git hooks

From the repo root:

```bash
pre-commit install
```

(Optional) validate setup on all files:

```bash
pre-commit run --all-files
```

## Useful commands

Run Ruff lint + auto-fix manually:

```bash
ruff check . --fix
```

Run Ruff formatter manually:

```bash
ruff format .
```

Run pre-commit manually on all files:

```bash
pre-commit run --all-files
```

## Troubleshooting

- Commit fails after auto-fix:
  - This is expected with `--exit-non-zero-on-fix`.
  - Re-stage files and commit again.
- VS Code not formatting:
  - Check Ruff extension is installed and enabled.
  - Confirm workspace settings are active for this folder.
- Different results between machines:
  - Use the versions pinned in `requirements.txt`.

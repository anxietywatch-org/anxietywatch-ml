# Contributing to anxietywatch-ml

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, release-ready branch. Default branch. Protected. |
| `develop` | Integration branch for ongoing development. Protected. |
| `feature/*` | New functionality. Target: `develop`. |
| `fix/*` | Bug fixes. Target: `develop` (or `main` for hotfixes). |
| `chore/*` | Maintenance, tooling, docs. Target: `develop`. |
| `hotfix/*` | Urgent production fixes. Target: `main`, then sync to `develop`. |

## Normal Workflow

```bash
# Start from latest develop
git switch develop
git pull --ff-only

# Create feature branch
git switch -c feature/<short-description>

# Work: small coherent commits
git add -p
git commit -m "feat: add <what>"

# Push and open PR
git push -u origin feature/<short-description>
# Open PR via GitHub UI targeting develop
```

After review and CI green: merge via PR (squash or merge). Delete branch.

## Release Flow

```bash
# develop → main (release PR)
git switch develop
git pull --ff-only
git switch -c chore/release-vX.Y.Z
# version bump if needed
git push -u origin chore/release-vX.Y.Z
# Open PR targeting main
# After CI green + review: merge
# Tag release on main: git tag vX.Y.Z && git push origin vX.Y.Z
```

## Rules

- No direct push to `main` or `develop` (protected by branch rulesets).
- CI required: all checks must pass on PR.
- Minimum 1 approving review.
- Conversation resolution required.
- Force push blocked on protected branches.
- Branch deletion blocked on protected branches.
- No datasets / model binaries / secrets committed.
- Commits: small, coherent, conventional messages (feat/fix/chore/docs).

## Local Validation (run before push)

```bash
# Install dev dependencies
python -m pip install -e ".[dev]"

# 1. Tests (164+)
python -m pytest tests/ -q

# 2. Ruff quality gate (exact CI command)
python -m ruff check src tests --config .ruff-ci.toml

# 3. Security audit (pip-audit)
python -m pip_audit

# 4. Package import smoke
python -m compileall -q src

# 5. Git hygiene
git diff --check
```

## CI Checks

The ML CI workflow (`.github/workflows/ci.yml`) runs on every push/PR to `main`/`develop`:

1. **quality-and-test** — Ruff gate + compileall + pytest (164+ tests)
2. **security-audit** — pip-audit (known vulnerability PYSEC-2026-1845 in pytest<9 is documented + ignored)

Required status check: `ML CI / quality-and-test`

## Ruff Quality Gate

The gate uses `.ruff-ci.toml` (strict clean subset + documented debt):

```bash
# Run exactly as CI
python -m ruff check src tests --config .ruff-ci.toml
```

Debt codes temporarily ignored are listed in `.ruff-ci.toml` with counts. To harden: fix violations and remove from `ignore`.

## Security

- Dependabot: weekly updates for pip + github-actions (5 PR limit).
- pip-audit runs in CI (skips PYSEC-2026-1845 in pytest 8.x — constraint `<9` prevents fix).
- Secret scanning + push protection: enabled via GitHub.
- No secrets in repo; use GitHub secrets for CI.

## Artifacts & Hygiene

- Models (`*.pkl`, `*.joblib`) → ignored via `.gitignore` (`/models/*.pkl`).
- Generated data → `/data/generated/` ignored.
- Coverage outputs → `.coverage`, `htmlcov/` ignored.
- IDE/OS junk → ignored.

## Local Development

```bash
# Create venv (optional but recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install
python -m pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -q

# Run inference service locally
python -c "from anxietywatch_ml.serving import train_demo_model; train_demo_model(output_path='models/prototype_v0.1.0.pkl')"
python -m uvicorn anxietywatch_ml.serving.app:app --port 8000
```

## Questions?

Open an issue or ask in the PR.
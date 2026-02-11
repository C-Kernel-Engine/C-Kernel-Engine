# Contributing to CK-Engine

## Git Workflow (v6.6+)

All code changes must go through branches and pass tests before merging to `main`.

### Branch Naming Convention

```
feature/  - New features (e.g., feature/fp16-kv-cache)
fix/      - Bug fixes (e.g., fix/q6k-avx2-symbol)
optimize/ - Performance improvements (e.g., optimize/q5_0-kernels)
refactor/ - Code restructuring (e.g., refactor/kernel-api)
docs/     - Documentation only (e.g., docs/mega-fusion)
```

### Development Workflow

```bash
# 1. Create a branch from main
git checkout main
git pull origin main
git checkout -b fix/my-fix-name

# 2. Make your changes
# ... edit files ...

# 3. Test your changes
make clean && make
make e2e                    # Must pass

# 4. Run pre-merge checks
./scripts/pre-merge-check.sh

# 5. Commit with descriptive message
git add <files>
git commit -m "Fix: description of what was fixed

- Detail 1
- Detail 2"

# 6. Push branch and create PR (or merge locally)
git push origin fix/my-fix-name

# 7. After review/CI passes, merge to main
git checkout main
git merge fix/my-fix-name
git push origin main
```

### Enable Git Hooks (Required)
This repo ships pre-push checks in `.githooks/pre-push`. Git does **not** enable repo hooks automatically on clone/pull.

Run once after cloning:

```bash
./scripts/setup-hooks.sh
```

Verify:

```bash
git config core.hooksPath
```

It should print `.githooks`. If not, pushes will **not** run the pre-checks/e2e.

### Pre-Merge Requirements

Before merging to `main`, ALL of these must pass:

| Check | Command | Required |
|-------|---------|----------|
| Build | `make` | Yes |
| E2E Inference | `make e2e` | Yes |
| Kernel Parity | `make test-parity` | Recommended |
| Pipeline Tests | `make test-pipeline-quick` | Recommended |

### Quick Check Commands

```bash
# Full pre-merge validation
./scripts/pre-merge-check.sh

# Quick check (skips slow tests)
./scripts/pre-merge-check.sh --quick

# Individual test suites
make e2e                    # End-to-end inference
make test-pipeline-quick    # 6-layer pipeline validation
make test-parity            # Kernel accuracy vs llama.cpp
```

### Test Pyramid (6 Layers)

When debugging failures, tests are ordered from low-level to high-level:

1. **Kernel Parity** - Do individual kernels match llama.cpp?
2. **Bump Conversion** - Are weights converted correctly?
3. **IR Validation** - Is the computation graph correct?
4. **Codegen** - Does generated C code compile?
5. **Tensor Flow** - Are dimensions correct throughout?
6. **E2E Inference** - Does it produce coherent output?

Run specific layers:
```bash
./scripts/test_full_pipeline.sh --layer 3  # Run only IR validation
```

## Code Style

- C code: Follow existing kernel style (see `src/kernels/`)
- Python: Black formatter, 100 char line width
- Commits: Imperative mood ("Fix bug" not "Fixed bug")

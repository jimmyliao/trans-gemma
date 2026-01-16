# 🚀 Quick Start with `uv`

This guide shows you how to run TranslateGemma examples using `uv`, the ultra-fast Python package manager.

## Prerequisites

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # or
   brew install uv
   ```

2. **Get a Hugging Face Token**:
   - Visit https://huggingface.co/settings/tokens
   - Create a new token with "Read" permissions
   - Apply for access to TranslateGemma: https://huggingface.co/google/translategemma-4b-it

---

## 🎯 Running Examples

### 1. Setup Environment

```bash
# Navigate to project directory
cd trans-gemma

# Create .env file from template
cp .env.example .env

# Edit .env and add your HF_TOKEN
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 2. Run Examples with `uv`

#### Option A: Quick Test (Verify HF Token)

```bash
# Install dependencies and run
uv run --extra examples examples/verify-hf-token.py
```

Or with environment variable:

```bash
HF_TOKEN="hf_xxxxx" uv run --extra examples examples/verify-hf-token.py
```

#### Option B: Full Local Test

```bash
# This will:
# 1. Load .env file
# 2. Verify HF_TOKEN
# 3. Test model access
# 4. Optionally test translation (requires GPU)

uv run --extra examples examples/local-test.py
```

#### Option C: TranslateGemma Fixed Format Example

```bash
# Shows correct usage of structured chat template
HF_TOKEN="hf_xxxxx" uv run --extra examples examples/translategemma-fix.py
```

---

## 📦 Dependency Management

### Install Specific Dependency Groups

```bash
# Install base dependencies only
uv sync

# Install with examples dependencies
uv sync --extra examples

# Install with Cloud Run dependencies
uv sync --extra cloudrun

# Install everything (examples + cloudrun + dev)
uv sync --extra all
```

### Run Scripts Without Installing

```bash
# uv run will automatically manage dependencies
uv run --extra examples python examples/local-test.py
```

---

## 🔧 Development Workflow

### 1. Setup Development Environment

```bash
# Install all dependencies including dev tools
uv sync --extra dev

# Or use uv run for one-off commands
uv run --extra dev pytest tests/
```

### 2. Run Linters

```bash
# Format code with black
uv run --extra dev black .

# Lint with ruff
uv run --extra dev ruff check .
```

### 3. Interactive Development

```bash
# Start IPython
uv run --extra dev ipython

# Start Jupyter
uv run --extra dev jupyter notebook
```

---

## 🌐 Cloud Run Deployment

### Install Cloud Run Dependencies

```bash
uv sync --extra cloudrun
```

### Test FastAPI Server Locally

```bash
# Set environment variables
export HF_TOKEN="hf_xxxxx"
export MODEL_ID="google/translategemma-4b-it"

# Run the server
uv run --extra cloudrun uvicorn cloudrun.main:app --host 0.0.0.0 --port 8080
```

### Deploy to Cloud Run

```bash
cd cloudrun
./deploy.sh
```

---

## 💡 Tips & Tricks

### 1. Fast Dependency Installation

```bash
# uv is 10-100x faster than pip!
# It caches dependencies and uses parallel downloads

uv sync --extra examples  # Lightning fast! ⚡
```

### 2. Create Isolated Environments

```bash
# uv automatically creates virtual environments
# No need to manually activate/deactivate!

uv run python examples/verify-hf-token.py
# ^ This just works, no venv activation needed
```

### 3. Lock Dependencies

```bash
# Generate uv.lock for reproducible builds
uv lock

# Install from lock file
uv sync --frozen
```

### 4. Add New Dependencies

```bash
# Add a new dependency
uv add requests

# Add to specific group
uv add --optional examples python-dotenv
```

---

## 🆘 Troubleshooting

### Error: "HF_TOKEN not found"

```bash
# Option 1: Use .env file
cp .env.example .env
# Edit .env and add your token

# Option 2: Export environment variable
export HF_TOKEN="hf_xxxxx"

# Option 3: Pass as argument (for verify-hf-token.py)
uv run --extra examples python examples/verify-hf-token.py hf_xxxxx
```

### Error: "Access denied to model"

1. Visit https://huggingface.co/google/translategemma-4b-it
2. Click "Request access" (usually approved immediately)
3. Make sure your token has "Read" permissions

### Error: "No GPU available"

For local testing without GPU:
- Use Google Colab (free T4 GPU): Open `translategemma-colab.ipynb`
- Or skip the translation test in `local-test.py` (it will ask you)

---

## 📚 Example Scripts Overview

| Script | Purpose | Requires GPU |
|--------|---------|--------------|
| `verify-hf-token.py` | Verify HF token and model access | ❌ No |
| `local-test.py` | Full local testing workflow | ⚠️ Optional |
| `translategemma-fix.py` | Correct chat template usage | ✅ Yes |
| `simple-translation.py` | Cloud Run API client | ❌ No |

---

## 🎓 Next Steps

1. ✅ Verify your setup: `uv run --extra examples examples/verify-hf-token.py`
2. 🚀 Open Colab notebook: `translategemma-colab.ipynb`
3. ☁️ Deploy to Cloud Run: `cd cloudrun && ./deploy.sh`
4. 📖 Read full documentation: [README.md](README.md)

---

**Questions?** Open an issue: https://github.com/jimmyliao/trans-gemma/issues

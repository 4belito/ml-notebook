# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

This project uses `uv` with Python 3.11 and an editable install so notebooks can import from the repo directly:

```bash
uv sync
uv pip install -e .
```

## Linting

Ruff is the formatter and linter. Active rule sets: `E, F, I, B, UP, SIM, C4`.

```bash
uvx ruff check .
uvx ruff format .
```

Type checking uses Pyright in strict mode (configured via VS Code; `python.analysis.typeCheckingMode: strict`). Suppressions for unknown ML library types are intentional.

## Architecture

Each concept lives in a paired `.py` module + `.ipynb` notebook. The `.py` file contains the implementation; the notebook defines the math, tests it against PyTorch's `nn.*` equivalent, and shows both eval and training outputs.

### Module layout

```text
models/deep_learning/
  components/      # reusable building blocks (imported as mynn components)
  architectures/   # assembled models (MLP, Conv2d, TransformerEncoder*)
  initialization/  # weight initialization utilities
optimizers/        # custom optimizer implementations (SGD, Adam, AdaGrad, etc.)
loss_functions/    # loss function implementations
```

`models.deep_learning.architectures` is the top-level import alias used in notebooks (`import models.deep_learning.architectures as mynn`).

### Component → Architecture dependency

`TransformerEncoderLayer` composes `SelfAttention` (from `components/`), which wraps `nn.MultiheadAttention`. The custom `MultiheadAttention` in `components/` uses standard `nn.Linear` init (Xavier uniform + zeros bias), intentionally differing from PyTorch's packed-weight initialization.

### Tensor annotations

All `forward` methods are annotated with `jaxtyping`: `Float[Tensor, "b n c"]`. Dimension shorthands are declared in each module's docstring (`b`: batch, `n`: sequence length, `c`/`d_model`: embedding dim, `h`: heads, etc.).

### Testing pattern in notebooks

Custom implementations are verified against `torch.nn.*` by initializing both with the same seed:
-`init_seed` controls weight initialization
-`train_seed` controls dropout randomness

Eval outputs must match exactly. Activation classes are passed as `activation_cls: type[nn.Module]` (e.g. `nn.GELU`), not as strings.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
uv sync
uv pip install -e .
```

Python 3.11 (pinned).

## Linting

```bash
uvx ruff check .
uvx ruff format .
```

Active rule sets: `E, F, I, B, UP, SIM, C4`. Type checking uses Pyright in strict mode (`python.analysis.typeCheckingMode: strict`). Suppressions for untyped ML libraries are intentional.

## Architecture

Each concept lives in a paired `.py` module + `.ipynb` notebook. The `.py` file is the implementation; the notebook shows the math, verifies against `nn.*`, and demos training/eval.

### Import aliases

In notebooks:

- `import models.deep_learning.architectures as mynn` — assembled models
- `import models.deep_learning.components as comp` — reusable building blocks

In `.py` files inside `architectures/`, use `from models.deep_learning.components import ...` (absolute, not relative).

### Component layer (`models/deep_learning/components/`)

| Module | Contents |
| --- | --- |
| `attention/` | `MultiheadAttention`, `SelfAttention`, `create_causal_mask`, `create_random_mask` |
| `embeddings/` | `Embedding`, `SinusoidalPE`, `FourierPositionalEncoding`, `RotaryEmbedding` |
| `normalization/` | `BatchNorm1d`, `LayerNorm` |
| `activations/` | `ReLU`, `LeakyReLU`, `PReLU`, `ELU`, `SELU`, `GELU`, `Sigmoid`, `Tanh` |
| `linear/` | `Linear` |
| `regularization/` | `Dropout` |
| `residual/` | `ResidualConnection` |

### Architecture layer (`models/deep_learning/architectures/`)

- `mlp.py` — `MLP`
- `convolution/conv2d.py` — `conv2d`
- `transformer/encoder.py` — `TransformerEncoderLayer`, `TransformerEncoder`
- `transformer/decoder.py` — `TransformerDecoderLayer`, `TransformerDecoder`
- `transformer/transformer.py` — `Transformer` (full encoder-decoder)

### Known deviations from `nn.Transformer`

Key deviations (full list in `encoder.py` module docstring):

- Always **batch-first** `(b, n, c)`; `nn` default is seq-first.
- `activation_cls` takes a class (`nn.GELU`), not a string.
- Attention uses separate `q_proj/k_proj/v_proj` (vs. `nn`'s packed `in_proj_weight`); use `load_weights_from_torch_*` for exact weight comparison.
- `key_padding_mask` is merged into the attention mask upstream; boolean masks are not supported — use float `0/-inf` masks.

### Tensor annotations

All `forward` methods use `jaxtyping`: `Float[Tensor, "b n c"]`. Dimension shorthands: `b` batch, `n` sequence length, `c`/`d_model` embedding dim, `h` heads. `F722` is ignored in ruff — jaxtyping shape strings are not valid Python expressions.

### Testing pattern in notebooks

Custom implementations are verified against `torch.nn.*` by initializing both with the same seed:

- `init_seed` — controls weight initialization
- `train_seed` — controls dropout randomness

Activation classes are passed as `activation_cls: type[nn.Module]` (e.g. `nn.GELU`), not as strings.

## Translation task (`transformer/tasks/translation/`)

### Module structure

```text
translation/
  config.py      — Config dataclass; derived paths (model_folder, tokenizer files) as @property
  dataset.py     — TranslationHFDataset adapter, BilingualDataset, create_dataloaders, causal_mask
  tokenizer.py   — get_or_build_tokenizer (WordLevel, HF tokenizers library)
  model.py       — Translator: InputEmbedding → mynn.TransformerEncoder/Decoder → Projection
  helpers.py     — get_device
  __init__.py    — re-exports all public symbols; import as `trn`
```

### HuggingFace ↔ PyTorch adapter

HF `datasets.Dataset` is untyped and incompatible with `torch.utils.data.Dataset[T]`. `TranslationHFDataset` wraps it as a properly typed `Dataset[TranslationRow]`, exposing `__len__`, `__getitem__`, `__iter__`, `filter`, and a `load_dataset` classmethod. Never use the raw HF dataset directly — always go through `TranslationHFDataset`.

### Sequence filtering

Always filter the raw dataset **before** creating dataloaders. Long sequences cause `ValueError: Source/Target sentence is too long` inside `BilingualDataset.__getitem__`.

```python
raw_ds = raw_ds.filter(
    lambda x: (
        len(tokenizer_src.encode(x["translation"][CONFIG.src_lang]).ids) <= CONFIG.src_seq_len - 2
        and len(tokenizer_tgt.encode(x["translation"][CONFIG.tgt_lang]).ids) <= CONFIG.tgt_seq_len - 1
    )
)
```

### HuggingFace auth

`HF_TOKEN` in `.env` (gitignored). Call `load_dotenv()` at the top of the notebook before loading any dataset.

### gitignored artifacts

- `tokenizer_??.json` — trained tokenizer files
- `Helsinki-NLP/` — cached HF dataset files
- `runs/` — TensorBoard logs
- `weights_*/` — saved model checkpoints

### Notebook tooling

Relative imports (`.config`, `.dataset`) work in `.py` files but **not** in Jupyter — use the full absolute module path.

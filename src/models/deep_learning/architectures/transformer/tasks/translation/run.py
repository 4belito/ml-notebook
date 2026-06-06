# %% [markdown]
# # Translation Task
#
# [video](https://www.youtube.com/watch?v=ISNdQcPhsts)

# %%
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter

import models.deep_learning.architectures.transformer.tasks.translation as trn

load_dotenv()
device = trn.get_device()

CONFIG = trn.Config(
    batch_size=8,
    num_epochs=50,
    lr=1e-4,
    src_seq_len=350,
    tgt_seq_len=350,
    d_model=512,
    dropout=0.1,
    datasource="Helsinki-NLP/opus_books",
    src_lang="en",
    tgt_lang="es",
    model_basename="tmodel_",
)


# %%
def _build_writer(log_dir: str) -> SummaryWriter:
    """Prefer local node storage for TensorBoard events to avoid Lustre I/O issues."""
    user = os.environ.get("USER", "user")
    local_log_dir = Path("/tmp") / user / "ml-notebook" / "runs"
    local_log_dir.mkdir(parents=True, exist_ok=True)

    try:
        return SummaryWriter(str(local_log_dir / Path(log_dir).name))
    except OSError as exc:
        print(f"Local TensorBoard path unavailable ({exc}); falling back to {log_dir}")
        return SummaryWriter(log_dir)


writer = _build_writer(CONFIG.experiment_name)
Path(CONFIG.weights_folder).mkdir(parents=True, exist_ok=True)


# %%
raw_ds = trn.TranslationHFDataset.load_dataset(
    path=CONFIG.datasource,
    name=f"{CONFIG.src_lang}-{CONFIG.tgt_lang}",
    split="train",
)

# %% [markdown]
# ## Tokenization

# %%
src_file = Path(CONFIG.tokenizer_src_file)
tgt_file = Path(CONFIG.tokenizer_tgt_file)

for p in (src_file, tgt_file):
    # Clean up accidental directories created in earlier runs.
    if p.exists() and p.is_dir():
        shutil.rmtree(p)
    p.parent.mkdir(parents=True, exist_ok=True)

tokenizer_src = trn.get_or_build_tokenizer(src_file, raw_ds, CONFIG.src_lang)
tokenizer_tgt = trn.get_or_build_tokenizer(tgt_file, raw_ds, CONFIG.tgt_lang)

# %%
src_file = Path(CONFIG.tokenizer_src_file)
tgt_file = Path(CONFIG.tokenizer_tgt_file)
src_file.parent.mkdir(parents=True, exist_ok=True)
tokenizer_src = trn.get_or_build_tokenizer(src_file, raw_ds, CONFIG.src_lang)
tokenizer_tgt = trn.get_or_build_tokenizer(tgt_file, raw_ds, CONFIG.tgt_lang)

# %% [markdown]
# ## Create dataloaders

# %%
train_dataloader, val_dataloader = trn.create_dataloaders(
    raw_ds, tokenizer_src, tokenizer_tgt, CONFIG
)

# %% [markdown]
# ## Create model

# %%
model = trn.Translator(
    src_vocab_size=tokenizer_src.get_vocab_size(),
    tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
    dropout=CONFIG.dropout,
    src_max_length=CONFIG.src_seq_len,
    tgt_max_length=CONFIG.tgt_seq_len,
    embed_size=CONFIG.d_model,
).to(device)

# %% [markdown]
# ## Train the model

# %%
trn.train(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    tokenizer_src=tokenizer_src,
    tokenizer_tgt=tokenizer_tgt,
    device=device,
    config=CONFIG,
    writer=writer,
)

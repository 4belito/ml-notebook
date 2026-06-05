# %% [markdown]
# # Translation Task
#
# [video](https://www.youtube.com/watch?v=ISNdQcPhsts)

# %%
from pathlib import Path

from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter

import models.deep_learning.architectures.transformer.tasks.translation as trn

load_dotenv()
device = trn.get_device()

# %% [markdown]
# ## Config

# %%
CONFIG = trn.Config(
    batch_size=8,
    num_epochs=50,
    lr=1e-4,
    src_seq_len=350,
    tgt_seq_len=350,
    d_model=512,
    datasource="Helsinki-NLP/opus_books",
    src_lang="en",
    tgt_lang="es",
    model_basename="tmodel_",
)


# %%
writer = SummaryWriter(CONFIG.experiment_name)
Path(CONFIG.model_folder).mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load Dataset (from HuggingFace)

# %%
raw_ds = trn.TranslationHFDataset.load_dataset(
    path=CONFIG.datasource,
    name=f"{CONFIG.src_lang}-{CONFIG.tgt_lang}",
    split="train",
)

# %% [markdown]
# ## Tokenization

# %%
tokenizer_src = trn.get_or_build_tokenizer(
    Path(CONFIG.tokenizer_src_file), raw_ds, CONFIG.src_lang
)
tokenizer_tgt = trn.get_or_build_tokenizer(
    Path(CONFIG.tokenizer_tgt_file), raw_ds, CONFIG.tgt_lang
)

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

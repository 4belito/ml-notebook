# RUN
# There is no arguments handled by argparse, this is just a version of the
# training part of the translation notebook that can be run as a script.
# To run:
# cd ~/ml-notebook/src
# python -m models.deep_learning.architectures.transformer.tasks.translation.run_train

import shutil
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

import models.deep_learning.architectures.transformer.tasks.translation as trn
from helpers import build_writer, get_device, setup_ddp

DDP_training = True
CONFIG = trn.Config(
    batch_size=8,
    num_epochs=5,
    lr=1e-4,
    src_seq_len=350,
    tgt_seq_len=350,
    d_model=512,
    dropout=0.1,
    datasource="Helsinki-NLP/opus_books",
    src_lang="en",
    tgt_lang="es",
    model_basename="testmodel2_",
)

load_dotenv()
rank, local_rank, world_size = 0, 0, 1
if DDP_training:
    rank, local_rank, world_size = setup_ddp()
    DDP_training = world_size > 1

device = torch.device(f"cuda:{local_rank}") if DDP_training else get_device()


writer = build_writer(CONFIG.experiment_name) if rank == 0 else None

## Data Preparation
if not DDP_training or rank == 0:
    Path(CONFIG.weights_folder).mkdir(parents=True, exist_ok=True)

# All ranks load the dataset (HF caches it locally, no redundant downloads)
raw_ds = trn.TranslationHFDataset.load_dataset(
    path=CONFIG.datasource,
    name=f"{CONFIG.src_lang}-{CONFIG.tgt_lang}",
    split="train",
)

src_file = Path(CONFIG.tokenizer_src_file)
tgt_file = Path(CONFIG.tokenizer_tgt_file)

if not DDP_training or rank == 0:
    for p in (src_file, tgt_file):
        # Clean up accidental directories created in earlier runs.
        if p.exists() and p.is_dir():
            shutil.rmtree(p)
        p.parent.mkdir(parents=True, exist_ok=True)
    # Rank 0 builds tokenizer files; other ranks wait at the barrier below
    trn.get_or_build_tokenizer(src_file, raw_ds, CONFIG.src_lang)
    trn.get_or_build_tokenizer(tgt_file, raw_ds, CONFIG.tgt_lang)

if DDP_training:
    # guarantee tokenizer files exist before all ranks load them
    torch.distributed.barrier()

tokenizer_src = trn.get_or_build_tokenizer(src_file, raw_ds, CONFIG.src_lang)
tokenizer_tgt = trn.get_or_build_tokenizer(tgt_file, raw_ds, CONFIG.tgt_lang)


train_ds, val_ds = trn.create_bilingual_datasets(
    raw_ds, tokenizer_src, tokenizer_tgt, CONFIG, verbose=(rank == 0)
)

train_sampler = (
    DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    if DDP_training
    else None
)
train_dloader, val_dloader = trn.create_dataloaders(
    CONFIG.batch_size, train_ds, val_ds, sampler=train_sampler
)

model = trn.Translator(
    src_vocab_size=tokenizer_src.get_vocab_size(),
    tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
    dropout=CONFIG.dropout,
    src_max_length=CONFIG.src_seq_len,
    tgt_max_length=CONFIG.tgt_seq_len,
    embed_size=CONFIG.d_model,
).to(device)


if DDP_training:
    model = DDP(model, device_ids=[local_rank])

trn.train(
    model=model,
    train_dataloader=train_dloader,
    val_dataloader=val_dloader,
    tokenizer_src=tokenizer_src,
    tokenizer_tgt=tokenizer_tgt,
    config=CONFIG,
    device=device,
    rank=rank,
    train_sampler=train_sampler,
    writer=writer,
)

if DDP_training:
    torch.distributed.destroy_process_group()

# RUN (DDP version for SMC cluster)
# Each srun task maps to one GPU via SLURM_LOCALID.
# Launch command (from ~/ml-notebook/src):
#
#   srun --nodes=1 --ntasks-per-node=8 python -m \
#       models.deep_learning.architectures.transformer.tasks.translation.run_train_ddp
#
# Slurm sets these automatically per task:
#   SLURM_PROCID   – global rank (0 … world_size-1)
#   SLURM_LOCALID  – GPU index on this node (0 … 7)
#   SLURM_NTASKS   – total tasks = world size

import os
import shutil
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from dotenv import load_dotenv
from tokenizers import Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.utils.tensorboard import SummaryWriter

import models.deep_learning.architectures.transformer.tasks.translation as trn
from models.deep_learning.architectures.transformer.tasks.translation.dataset import (
    BilingualDataset,
    TranslationHFDataset,
)
from models.deep_learning.architectures.transformer.tasks.translation.train_ddp import (
    train_ddp,
)

from .dataset import BilingualSample


def _setup_ddp() -> tuple[int, int, int]:
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    if "MASTER_ADDR" not in os.environ:
        # All tasks run on the same node, so hostname is always correct here.
        # For multi-node jobs, set MASTER_ADDR explicitly before calling srun.
        os.environ["MASTER_ADDR"] = socket.gethostname()
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _create_ddp_dataloaders(
    raw_ds: TranslationHFDataset,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    config: trn.Config,
    rank: int,
    world_size: int,
) -> tuple[
    DataLoader[BilingualSample],
    DataLoader[BilingualSample],
    DistributedSampler[BilingualSample],
]:
    raw_ds = raw_ds.filter(
        lambda x: (
            len(tokenizer_src.encode(x["translation"][config.src_lang]).ids)
            <= config.src_seq_len - 2
            and len(tokenizer_tgt.encode(x["translation"][config.tgt_lang]).ids)
            <= config.tgt_seq_len - 1
        )
    )

    train_ds_size = int(config.train_size * len(raw_ds))
    val_ds_size = len(raw_ds) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(raw_ds, [train_ds_size, val_ds_size])

    shared = {
        "tokenizer_src": tokenizer_src,
        "tokenizer_tgt": tokenizer_tgt,
        "src_lang": config.src_lang,
        "tgt_lang": config.tgt_lang,
        "src_seq_len": config.src_seq_len,
        "tgt_seq_len": config.tgt_seq_len,
    }
    train_ds = BilingualDataset(train_ds_raw, **shared)
    val_ds = BilingualDataset(val_ds_raw, **shared)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_dataloader: DataLoader[dict[str, torch.Tensor | str]] = DataLoader(
        train_ds, batch_size=config.batch_size, sampler=train_sampler
    )
    # val is only used by rank 0; other ranks skip it
    val_dataloader: DataLoader[dict[str, torch.Tensor | str]] = DataLoader(
        val_ds, batch_size=1, shuffle=False
    )

    return train_dataloader, val_dataloader, train_sampler


def _build_writer(log_dir: str) -> SummaryWriter:
    user = os.environ.get("USER", "user")
    local_log_dir = Path("/tmp") / user / "ml-notebook" / "runs"
    local_log_dir.mkdir(parents=True, exist_ok=True)
    try:
        return SummaryWriter(str(local_log_dir / Path(log_dir).name))
    except OSError as exc:
        print(f"Local TensorBoard path unavailable ({exc}); falling back to {log_dir}")
        return SummaryWriter(log_dir)


load_dotenv()

rank, local_rank, world_size = _setup_ddp()
device = torch.device(f"cuda:{local_rank}")

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

writer = _build_writer(CONFIG.experiment_name) if rank == 0 else None
if rank == 0:
    Path(CONFIG.weights_folder).mkdir(parents=True, exist_ok=True)

raw_ds = trn.TranslationHFDataset.load_dataset(
    path=CONFIG.datasource,
    name=f"{CONFIG.src_lang}-{CONFIG.tgt_lang}",
    split="train",
)

src_file = Path(CONFIG.tokenizer_src_file)
tgt_file = Path(CONFIG.tokenizer_tgt_file)

if rank == 0:
    for p in (src_file, tgt_file):
        if p.exists() and p.is_dir():
            shutil.rmtree(p)
        p.parent.mkdir(parents=True, exist_ok=True)
    # Rank 0 builds tokenizer files; other ranks wait at the barrier below
    trn.get_or_build_tokenizer(src_file, raw_ds, CONFIG.src_lang)
    trn.get_or_build_tokenizer(tgt_file, raw_ds, CONFIG.tgt_lang)

dist.barrier()  # guarantee tokenizer files exist before all ranks load them

tokenizer_src = trn.get_or_build_tokenizer(src_file, raw_ds, CONFIG.src_lang)
tokenizer_tgt = trn.get_or_build_tokenizer(tgt_file, raw_ds, CONFIG.tgt_lang)

train_dataloader, val_dataloader, train_sampler = _create_ddp_dataloaders(
    raw_ds, tokenizer_src, tokenizer_tgt, CONFIG, rank, world_size
)

model = trn.Translator(
    src_vocab_size=tokenizer_src.get_vocab_size(),
    tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
    dropout=CONFIG.dropout,
    src_max_length=CONFIG.src_seq_len,
    tgt_max_length=CONFIG.tgt_seq_len,
    embed_size=CONFIG.d_model,
).to(device)

model = DDP(model, device_ids=[local_rank])

train_ddp(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    tokenizer_src=tokenizer_src,
    tokenizer_tgt=tokenizer_tgt,
    device=device,
    config=CONFIG,
    writer=writer,
    rank=rank,
    train_sampler=train_sampler,
)

dist.destroy_process_group()

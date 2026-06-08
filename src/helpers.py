import os
import socket
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


def get_device() -> torch.device:

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def clear_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def setup_ddp() -> tuple[int, int, int]:
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    if "MASTER_ADDR" not in os.environ:
        # All tasks run on the same node, so hostname is always correct here.
        # For multi-node jobs, set MASTER_ADDR explicitly before calling srun.
        os.environ["MASTER_ADDR"] = socket.gethostname()
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def build_writer(log_dir: str) -> SummaryWriter:
    user = os.environ.get("USER", "user")
    local_log_dir = Path("/tmp") / user / "ml-notebook" / "runs"
    local_log_dir.mkdir(parents=True, exist_ok=True)
    try:
        return SummaryWriter(str(local_log_dir / Path(log_dir).name))
    except OSError as exc:
        print(f"Local TensorBoard path unavailable ({exc}); falling back to {log_dir}")
        return SummaryWriter(log_dir)

from dataclasses import dataclass
from pathlib import Path

from config import DATA_DIR, RUNS_DIR, WEIGHTS_DIR


@dataclass
class Config:
    batch_size: int
    num_epochs: int
    lr: float
    src_seq_len: int
    tgt_seq_len: int
    d_model: int
    dropout: float
    datasource: str
    src_lang: str
    tgt_lang: str
    preload: str = "latest"
    model_basename: str = "tmodel_"
    train_size: float = 0.9

    @property
    def weights_folder(self) -> str:
        return (
            f"{WEIGHTS_DIR}/{self.datasource}/{self.model_basename}"
            f"_{self.src_lang}_{self.tgt_lang}"
        )

    @property
    def experiment_name(self) -> str:
        return (
            f"{RUNS_DIR}/{self.datasource}/{self.model_basename}"
            f"_{self.src_lang}_{self.tgt_lang}"
        )

    @property
    def tokenizer_src_file(self) -> str:
        return f"{DATA_DIR}/cache/{self.datasource}/tokenizer_{self.src_lang}.json"

    @property
    def tokenizer_tgt_file(self) -> str:
        return f"{DATA_DIR}/cache/{self.datasource}/tokenizer_{self.tgt_lang}.json"

    def get_weights_file_path(self, epoch: str) -> str:
        return f"{self.weights_folder}/{self.model_basename}{epoch}.pt"

    def latest_weights_file_path(self) -> str | None:
        weights_files = list(Path(self.weights_folder).glob(f"{self.model_basename}*"))
        if len(weights_files) == 0:
            return None
        weights_files.sort()
        return str(weights_files[-1])

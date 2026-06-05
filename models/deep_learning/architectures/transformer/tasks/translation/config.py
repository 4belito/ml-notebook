from dataclasses import dataclass
from pathlib import Path


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
    def model_folder(self) -> str:
        return f"{self.datasource}_weights_{self.src_lang}_{self.tgt_lang}"

    @property
    def experiment_name(self) -> str:
        return f"runs/tmodel_{self.src_lang}_{self.tgt_lang}"

    @property
    def tokenizer_src_file(self) -> str:
        return f"tokenizer_{self.src_lang}.json"

    @property
    def tokenizer_tgt_file(self) -> str:
        return f"tokenizer_{self.tgt_lang}.json"

    def get_weights_file_path(self, epoch: str) -> str:
        return f"{self.model_folder}/{self.model_basename}{epoch}.pt"

    def latest_weights_file_path(self) -> str | None:
        weights_files = list(Path(self.model_folder).glob(f"{self.model_basename}*"))
        if len(weights_files) == 0:
            return None
        weights_files.sort()
        return str(weights_files[-1])

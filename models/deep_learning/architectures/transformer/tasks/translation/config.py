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
    datasource: str
    src_lang: str
    tgt_lang: str
    preload: str = "latest"
    model_basename: str = "tmodel_"
    train_size: float = 0.9

    @property
    def model_folder(self) -> str:
        return f"weights_{self.src_lang}_{self.tgt_lang}"

    @property
    def experiment_name(self) -> str:
        return f"runs/tmodel_{self.src_lang}_{self.tgt_lang}"

    @property
    def tokenizer_src_file(self) -> str:
        return f"tokenizer_{self.src_lang}.json"

    @property
    def tokenizer_tgt_file(self) -> str:
        return f"tokenizer_{self.tgt_lang}.json"


def get_weights_file_path(config: Config, epoch: str) -> str:
    model_folder = f"{config.datasource}_{config.model_folder}"
    model_filename = f"{config.model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config: Config) -> str | None:
    model_folder = f"{config.datasource}_{config.model_folder}"
    model_filename = f"{config.model_basename}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

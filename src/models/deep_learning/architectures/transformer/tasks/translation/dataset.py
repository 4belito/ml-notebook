from __future__ import annotations

from collections.abc import Callable, Iterator

import torch
from datasets import Dataset as RawHFDataset  # type: ignore[import-untyped]
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    Subset,
    random_split,
)

from .config import Config

TranslationRow = dict[str, dict[str, str]]
BilingualSample = dict[str, torch.Tensor | str]


class TranslationHFDataset(Dataset[TranslationRow]):
    def __init__(self, hf_ds: RawHFDataset) -> None:
        self._ds = hf_ds

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> TranslationRow:
        return self._ds[idx]

    def __iter__(self) -> Iterator[TranslationRow]:
        yield from self._ds  # type: ignore[misc]

    def filter(self, fn: Callable[[TranslationRow], bool]) -> TranslationHFDataset:
        return TranslationHFDataset(self._ds.filter(fn))

    @classmethod
    def load_dataset(cls, path: str, name: str, split: str) -> TranslationHFDataset:
        return cls(
            load_dataset(
                path=path,
                name=name,
                split=split,
            )
        )


class BilingualDataset(Dataset[BilingualSample]):
    def __init__(
        self,
        ds: Dataset[TranslationRow],
        tokenizer_src: Tokenizer,
        tokenizer_tgt: Tokenizer,
        src_lang: str,
        tgt_lang: str,
        src_seq_len: int,
        tgt_seq_len: int,
    ):
        super().__init__()
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self) -> int:
        return len(self.ds)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> BilingualSample:
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        src_input_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        src_num_padding_tokens = self.src_seq_len - len(src_input_tokens) - 2
        tgt_num_padding_tokens = self.tgt_seq_len - len(tgt_input_tokens) - 1

        if src_num_padding_tokens < 0:
            raise ValueError("Source sentence is too long")
        if tgt_num_padding_tokens < 0:
            raise ValueError("Target sentence is too long")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(src_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.full(
                    (src_num_padding_tokens,), self.pad_token.item(), dtype=torch.int64
                ),
            ],
            dim=0,
        )
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(tgt_input_tokens, dtype=torch.int64),
                torch.full(
                    (tgt_num_padding_tokens,), self.pad_token.item(), dtype=torch.int64
                ),
            ],
            dim=0,
        )
        label = torch.cat(
            [
                torch.tensor(tgt_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.full(
                    (tgt_num_padding_tokens,), self.pad_token.item(), dtype=torch.int64
                ),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.src_seq_len
        assert decoder_input.size(0) == self.tgt_seq_len
        assert label.size(0) == self.tgt_seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input == self.pad_token).unsqueeze(0).unsqueeze(0),
            "decoder_mask": (decoder_input == self.pad_token).unsqueeze(0)
            | causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def _filter_by_length(
    raw_ds: TranslationHFDataset,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    config: Config,
    verbose: bool = True,
) -> TranslationHFDataset:
    """Drop rows whose token length would overflow BilingualDataset's fixed seq_len."""
    if verbose:
        original_len = len(raw_ds)

    raw_ds = raw_ds.filter(
        lambda x: (
            len(tokenizer_src.encode(x["translation"][config.src_lang]).ids)
            <= config.src_seq_len - 2
            and len(tokenizer_tgt.encode(x["translation"][config.tgt_lang]).ids)
            <= config.tgt_seq_len - 1
        )
    )

    if verbose:
        filtered_len = len(raw_ds)
        removal_percentage = (original_len - filtered_len) / original_len * 100
        print(f"Original dataset size: {original_len}")
        print(f"Filtered dataset size: {filtered_len}")
        print(
            f"Removed: {original_len - filtered_len} samples "
            f"({removal_percentage:.2f}%)"
        )

    return raw_ds


def split_train_val(
    raw_ds: TranslationHFDataset, train_size: float
) -> tuple[Subset[TranslationRow], Subset[TranslationRow]]:
    train_ds_size = int(train_size * len(raw_ds))
    val_ds_size = len(raw_ds) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(raw_ds, [train_ds_size, val_ds_size])
    return train_ds_raw, val_ds_raw


def causal_mask(size: int) -> torch.Tensor:
    # True in upper triangle = block future positions (nn.MultiheadAttention convention)
    return torch.triu(torch.ones((1, size, size), dtype=torch.bool), diagonal=1)


def create_bilingual_datasets(
    raw_ds: TranslationHFDataset,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    config: Config,
    verbose: bool = True,
) -> tuple[Dataset[BilingualSample], Dataset[BilingualSample]]:
    raw_ds = _filter_by_length(
        raw_ds, tokenizer_src, tokenizer_tgt, config, verbose=verbose
    )
    train_ds_raw, val_ds_raw = split_train_val(raw_ds, config.train_size)

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

    return train_ds, val_ds


def create_dataloaders(
    batch_size: int,
    train_ds: Dataset[BilingualSample],
    val_ds: Dataset[BilingualSample],
    sampler: DistributedSampler[BilingualSample] | None = None,
) -> tuple[DataLoader[BilingualSample], DataLoader[BilingualSample]]:
    # When a sampler is provided (DDP), shuffle must be False — the sampler handles it.
    train_dataloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader

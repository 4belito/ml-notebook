from __future__ import annotations

from collections.abc import Callable, Iterator

import torch
from datasets import Dataset as RawHFDataset  # type: ignore[import-untyped]
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset, random_split

from .config import Config

TranslationRow = dict[str, dict[str, str]]


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


class BilingualDataset(Dataset[dict[str, torch.Tensor | str]]):
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

    def __getitem__(self, idx: int):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Transform the text into tokens
        src_input_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        src_num_padding_tokens = (
            self.src_seq_len - len(src_input_tokens) - 2
        )  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        tgt_num_padding_tokens = self.tgt_seq_len - len(tgt_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence
        # is too long
        if src_num_padding_tokens < 0:
            raise ValueError("Source sentence is too long")
        if tgt_num_padding_tokens < 0:
            raise ValueError("Target sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(src_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * src_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(tgt_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * tgt_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(tgt_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * tgt_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.src_seq_len
        assert decoder_input.size(0) == self.tgt_seq_len
        assert label.size(0) == self.tgt_seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input == self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0),  # (1, 1, seq_len) — True=block
            "decoder_mask": (decoder_input == self.pad_token).unsqueeze(0)
            | causal_mask(decoder_input.size(0)),  # (1, seq_len, seq_len) — True=block
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def create_dataloaders(
    raw_ds: TranslationHFDataset,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    config: Config,
):
    train_ds_size = int(config.train_size * len(raw_ds))
    val_ds_size = len(raw_ds) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(raw_ds, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config.src_lang,
        config.tgt_lang,
        config.src_seq_len,
        config.tgt_seq_len,
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config.src_lang,
        config.tgt_lang,
        config.src_seq_len,
        config.tgt_seq_len,
    )

    train_dataloader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader


def causal_mask(size: int) -> torch.Tensor:
    # True in upper triangle = block future positions (nn.MultiheadAttention convention)
    return torch.triu(torch.ones((1, size, size), dtype=torch.bool), diagonal=1)

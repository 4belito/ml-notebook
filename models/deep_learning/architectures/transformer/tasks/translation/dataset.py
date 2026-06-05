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
    ):
        super().__init__()
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

    def __len__(self) -> int:
        return len(self.ds)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        src_input_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(src_input_tokens, dtype=torch.int64),
                self.eos_token,
            ]
        )
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(tgt_input_tokens, dtype=torch.int64),
            ]
        )
        label = torch.cat(
            [
                torch.tensor(tgt_input_tokens, dtype=torch.int64),
                self.eos_token,
            ]
        )

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def _make_collate_fn(
    pad_id: int,
) -> Callable[
    [list[dict[str, torch.Tensor | str]]], dict[str, torch.Tensor | list[str]]
]:
    def collate_fn(
        batch: list[dict[str, torch.Tensor | str]],
    ) -> dict[str, torch.Tensor | list[str]]:
        src_max = max(item["encoder_input"].size(0) for item in batch)  # type: ignore[union-attr]
        tgt_max = max(item["decoder_input"].size(0) for item in batch)  # type: ignore[union-attr]

        encoder_inputs, decoder_inputs, labels = [], [], []
        encoder_masks, decoder_masks = [], []
        src_texts: list[str] = []
        tgt_texts: list[str] = []
        tgt_causal_mask = causal_mask(tgt_max)

        for item in batch:
            enc: torch.Tensor = item["encoder_input"]  # type: ignore[assignment]
            dec: torch.Tensor = item["decoder_input"]  # type: ignore[assignment]
            lbl: torch.Tensor = item["label"]  # type: ignore[assignment]

            enc_padded = torch.cat(
                [enc, torch.full((src_max - enc.size(0),), pad_id, dtype=torch.int64)]
            )
            dec_padded = torch.cat(
                [dec, torch.full((tgt_max - dec.size(0),), pad_id, dtype=torch.int64)]
            )
            lbl_padded = torch.cat(
                [lbl, torch.full((tgt_max - lbl.size(0),), pad_id, dtype=torch.int64)]
            )

            encoder_inputs.append(enc_padded)
            decoder_inputs.append(dec_padded)
            labels.append(lbl_padded)
            encoder_masks.append((enc_padded == pad_id).unsqueeze(0).unsqueeze(0))
            decoder_masks.append(
                (dec_padded == pad_id).unsqueeze(0) | tgt_causal_mask
            )
            src_texts.append(item["src_text"])  # type: ignore[arg-type]
            tgt_texts.append(item["tgt_text"])  # type: ignore[arg-type]

        return {
            "encoder_input": torch.stack(encoder_inputs),
            "decoder_input": torch.stack(decoder_inputs),
            "encoder_mask": torch.stack(encoder_masks),
            "decoder_mask": torch.stack(decoder_masks),
            "label": torch.stack(labels),
            "src_text": src_texts,
            "tgt_text": tgt_texts,
        }

    return collate_fn


def _filter_by_length(
    raw_ds: TranslationHFDataset,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    config: Config,
    verbose: bool = True,
) -> TranslationHFDataset:
    """Drop rows whose token length would overflow BilingualDataset's fixed seq_len."""
    original_len = len(raw_ds) if verbose else 0

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
        removed = original_len - filtered_len
        print(f"Original dataset size: {original_len}")
        print(f"Filtered dataset size: {filtered_len}")
        print(f"Removed: {removed} samples ({removed / original_len * 100:.2f}%)")

    return raw_ds


def create_dataloaders(
    raw_ds: TranslationHFDataset,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    config: Config,
    verbose: bool = True,
):
    raw_ds = _filter_by_length(
        raw_ds, tokenizer_src, tokenizer_tgt, config, verbose=verbose
    )
    train_ds_size = int(config.train_size * len(raw_ds))
    val_ds_size = len(raw_ds) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(raw_ds, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config.src_lang,
        config.tgt_lang,
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config.src_lang,
        config.tgt_lang,
    )

    pad_id = tokenizer_tgt.token_to_id("[PAD]")
    if pad_id is None:
        raise
    collate_fn = _make_collate_fn(pad_id)

    train_dataloader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_ds, batch_size=1, shuffle=True, collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader


def causal_mask(size: int) -> torch.Tensor:
    # True in upper triangle = block future positions (nn.MultiheadAttention convention)
    return torch.triu(torch.ones((1, size, size), dtype=torch.bool), diagonal=1)

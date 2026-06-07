from __future__ import annotations

import shutil
from collections.abc import Callable

import torch
import torchmetrics.text
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]

from .dataset import causal_mask
from .model import Translator

cer_metric = torchmetrics.text.CharErrorRate()
wer_metric = torchmetrics.text.WordErrorRate()
bleu_metric = torchmetrics.text.BLEUScore()


def _build_source_inputs(
    text: str,
    tokenizer_src: Tokenizer,
    src_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_pad_idx = tokenizer_src.token_to_id("[PAD]")
    src_sos_idx = tokenizer_src.token_to_id("[SOS]")
    src_eos_idx = tokenizer_src.token_to_id("[EOS]")

    if src_pad_idx is None or src_sos_idx is None or src_eos_idx is None:
        raise ValueError("Source tokenizer is missing one of [PAD], [SOS], [EOS].")

    src_ids = tokenizer_src.encode(text).ids
    if len(src_ids) > src_seq_len - 2:
        raise ValueError(
            f"Input is too long ({len(src_ids)} tokens). "
            f"Max allowed is {src_seq_len - 2}."
        )

    encoder_input = torch.full(
        (src_seq_len,), src_pad_idx, dtype=torch.int64, device=device
    )
    encoder_input[0] = src_sos_idx
    encoder_input[1 : 1 + len(src_ids)] = torch.tensor(
        src_ids, dtype=torch.int64, device=device
    )
    encoder_input[1 + len(src_ids)] = src_eos_idx

    source = encoder_input.unsqueeze(0)
    source_mask = (source == src_pad_idx).unsqueeze(1).unsqueeze(1)
    return source, source_mask


def greedy_decode(
    model: Translator,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_tgt: Tokenizer,
    tgt_max_len: int,
    device: torch.device,
) -> torch.Tensor:
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    if sos_idx is None or eos_idx is None:
        raise ValueError("Target tokenizer is missing one of [SOS], [EOS].")

    encoder_output = model.encode(source, source_mask)
    tokens: list[int] = [sos_idx]

    while len(tokens) < tgt_max_len:
        decoder_input = torch.tensor([tokens], dtype=source.dtype, device=device)
        decoder_mask = causal_mask(decoder_input.size(1)).unsqueeze(0).to(device)

        out = model.decode(decoder_input, encoder_output, decoder_mask, source_mask)
        prob = model.project(out[:, -1])
        next_word = int(torch.max(prob, dim=1).indices.item())
        tokens.append(next_word)

        if next_word == eos_idx:
            break

    return torch.tensor(tokens, dtype=source.dtype, device=device)


def translate_text(
    model: Translator,
    text: str,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    src_seq_len: int,
    tgt_max_len: int,
    device: torch.device,
) -> str:
    source, source_mask = _build_source_inputs(
        text=text,
        tokenizer_src=tokenizer_src,
        src_seq_len=src_seq_len,
        device=device,
    )

    with torch.no_grad():
        output_tokens = greedy_decode(
            model=model,
            source=source,
            source_mask=source_mask,
            tokenizer_tgt=tokenizer_tgt,
            tgt_max_len=tgt_max_len,
            device=device,
        )

    return tokenizer_tgt.decode(output_tokens.detach().cpu().tolist())


def run_validation(
    model: Translator,
    validation_ds: DataLoader[dict[str, torch.Tensor | str]],
    tokenizer_tgt: Tokenizer,
    tgt_max_len: int,
    device: torch.device,
    print_msg: Callable[[str], None],
    global_step: int,
    writer: SummaryWriter | None,
    num_examples: int = 2,
) -> None:
    model.eval()
    count = 0
    expected: list[str] = []
    predicted: list[str] = []

    console_width = shutil.get_terminal_size().columns

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model, encoder_input, encoder_mask, tokenizer_tgt, tgt_max_len, device
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(
                model_out.detach().cpu().numpy().tolist()
            )

            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg("-" * console_width)
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg("-" * console_width)
                break

    if writer:
        writer.add_scalar(
            "validation cer", cer_metric(predicted, expected), global_step
        )
        writer.add_scalar(
            "validation wer", wer_metric(predicted, expected), global_step
        )
        writer.add_scalar(
            "validation BLEU", bleu_metric(predicted, expected), global_step
        )
        writer.flush()

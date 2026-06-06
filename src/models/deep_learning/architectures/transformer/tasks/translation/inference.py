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
    # Both are guaranteed non-None by get_or_build_tokenizer validation
    assert sos_idx is not None
    assert eos_idx is not None

    encoder_output = model.encode(source, source_mask)
    tokens: list[int] = [sos_idx]

    while len(tokens) < tgt_max_len:
        decoder_input = torch.tensor([tokens], dtype=source.dtype, device=device)
        decoder_mask = causal_mask(decoder_input.size(1)).unsqueeze(0).to(device)

        out = model.decode(decoder_input, encoder_output, decoder_mask, source_mask)
        prob = model.project(out[:, -1])
        next_word = torch.max(prob, dim=1).indices.item()
        tokens.append(next_word)  # type: ignore[arg-type]

        if next_word == eos_idx:
            break

    return torch.tensor(tokens, dtype=source.dtype, device=device)


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

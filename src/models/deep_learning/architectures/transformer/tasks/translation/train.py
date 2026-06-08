from __future__ import annotations

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from .config import Config
from .dataset import BilingualSample
from .inference import run_validation
from .model import Translator


def train(
    model: Translator,
    train_dataloader: DataLoader[BilingualSample],
    val_dataloader: DataLoader[BilingualSample],
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    device: torch.device,
    config: Config,
    writer: SummaryWriter | None,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)

    initial_epoch = 0
    global_step = 0

    model_filename = (
        config.latest_weights_file_path()
        if config.preload == "latest"
        else config.get_weights_file_path(epoch=config.preload)
    )

    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No model to preload, starting from scratch")

    PAD_IDX = tokenizer_src.token_to_id("[PAD]")
    if PAD_IDX is None:
        raise ValueError("PAD token not found in the source tokenizer vocabulary")
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1).to(device)
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    writer_enabled = writer is not None

    def disable_writer(exc: Exception) -> None:
        nonlocal writer, writer_enabled
        if writer_enabled:
            print(f"TensorBoard disabled due to logging error: {exc}")
            writer_enabled = False
            writer = None

    for epoch in range(initial_epoch, config.num_epochs):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                decoder_input, encoder_output, decoder_mask, encoder_mask
            )
            proj_output = model.project(decoder_output)

            label = batch["label"].to(device)
            loss: torch.Tensor = loss_fn(
                proj_output.view(-1, tgt_vocab_size), label.view(-1)
            )

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            if writer_enabled and writer is not None:
                try:
                    writer.add_scalar("train loss", loss.item(), global_step)
                except (OSError, RuntimeError) as exc:
                    disable_writer(exc)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        if writer_enabled and writer is not None:
            try:
                writer.flush()
            except (OSError, RuntimeError) as exc:
                disable_writer(exc)

        try:
            run_validation(
                model,
                val_dataloader,
                tokenizer_tgt,
                config.tgt_seq_len,
                device,
                lambda msg: batch_iterator.write(msg),  # noqa: B023
                global_step,
                writer if writer_enabled else None,
            )
        except (OSError, RuntimeError) as exc:
            disable_writer(exc)
            run_validation(
                model,
                val_dataloader,
                tokenizer_tgt,
                config.tgt_seq_len,
                device,
                lambda msg: batch_iterator.write(msg),  # noqa: B023
                global_step,
                None,
            )

        model_filename = config.get_weights_file_path(f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )

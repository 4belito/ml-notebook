from __future__ import annotations

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from .config import Config
from .dataset import BilingualSample
from .inference import run_validation
from .model import Translator


def train(
    model: Translator | DDP,
    train_dataloader: DataLoader[BilingualSample],
    val_dataloader: DataLoader[BilingualSample],
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    device: torch.device,
    config: Config,
    writer: SummaryWriter | None,
    rank: int = 0,
    train_sampler: DistributedSampler[BilingualSample] | None = None,
) -> None:
    translator: Translator = model.module if isinstance(model, DDP) else model  # type: ignore[assignment]

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)

    initial_epoch = 0
    global_step = 0

    model_filename = (
        config.latest_weights_file_path()
        if config.preload == "latest"
        else config.get_weights_file_path(epoch=config.preload)
    )

    if model_filename:
        if rank == 0:
            print(f"Preloading model {model_filename}")
        state = torch.load(model_filename, map_location=device)
        translator.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        if rank == 0:
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
            if rank == 0:
                print(f"TensorBoard disabled due to logging error: {exc}")
            writer_enabled = False
            writer = None

    for epoch in range(initial_epoch, config.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Processing Epoch {epoch:02d}",
            disable=rank != 0,
        )
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            encoder_output = translator.encode(encoder_input, encoder_mask)
            decoder_output = translator.decode(
                decoder_input, encoder_output, decoder_mask, encoder_mask
            )
            proj_output = translator.project(decoder_output)

            label = batch["label"].to(device)
            loss: torch.Tensor = loss_fn(
                proj_output.view(-1, tgt_vocab_size), label.view(-1)
            )
            if rank == 0:
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

        if rank == 0 and writer_enabled and writer is not None:
            try:
                writer.flush()
            except (OSError, RuntimeError) as exc:
                disable_writer(exc)
        if rank == 0:
            try:
                run_validation(
                    translator,
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
                    translator,
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
                    "model_state_dict": translator.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                },
                model_filename,
            )

        if train_sampler is not None:
            torch.distributed.barrier()

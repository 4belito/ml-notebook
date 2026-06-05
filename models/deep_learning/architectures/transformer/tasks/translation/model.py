import math

import torch
import torch.nn as nn

import models.deep_learning.architectures as mynn
import models.deep_learning.components as comp


class InputEmbedding(nn.Module):
    def __init__(
        self, embed_size: int, vocab_size: int, max_length: int, dropout: float = 0.1
    ):
        super().__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = comp.SinusoidalPE(
            d_model=embed_size, max_len=max_length
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x) * math.sqrt(self.embed_size)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        return x


class Projection(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # CrossEntropyLoss applies log_softmax internally


class Translator(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_max_length: int = 100,
        tgt_max_length: int = 100,
        embed_size: int = 256,
        num_layers: int = 6,
        heads: int = 8,
        forward_dimension: int = 2048,
        dropout: float = 0.0,
        device: torch.device | None = None,
    ):
        super().__init__()

        # Encoder
        self.src_embedding = InputEmbedding(
            embed_size=embed_size,
            vocab_size=src_vocab_size,
            max_length=src_max_length,
            dropout=dropout,
        )
        self.encoder_layer = mynn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=heads,
            dim_feedforward=forward_dimension,
            dropout=dropout,
            device=device,
        )
        self.encoder = mynn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
        )

        # Decoder
        self.tgt_embedding = InputEmbedding(
            embed_size=embed_size,
            vocab_size=tgt_vocab_size,
            max_length=tgt_max_length,
            dropout=dropout,
        )
        self.decoder_layer = mynn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=heads,
            dim_feedforward=forward_dimension,
            dropout=dropout,
            device=device,
        )
        self.decoder = mynn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers,
        )

        # Final linear layer
        self.projection = Projection(d_model=embed_size, vocab_size=tgt_vocab_size)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.src_embedding(src), mask=src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(
            self.tgt_embedding(tgt),
            memory,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
        )

    def project(self, dec_output: torch.Tensor) -> torch.Tensor:
        return self.projection(dec_output)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        enc_src = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_src, tgt_mask, src_mask)
        return self.project(dec_output)

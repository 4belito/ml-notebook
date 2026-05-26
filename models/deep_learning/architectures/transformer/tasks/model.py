import math

import torch
import torch.nn as nn

import models.deep_learning.architectures as mynn
import models.deep_learning.components as comp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.word_embedding(x) * math.sqrt(self.d_model)


class Projection(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # raw logits — CrossEntropyLoss applies log_softmax internally


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
        self.src_embedding = nn.Sequential(
            InputEmbedding(d_model=embed_size, vocab_size=src_vocab_size),
            comp.SinusoidalPE(d_model=embed_size, max_len=src_max_length),
            nn.Dropout(0.1),
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
        self.tgt_embedding = nn.Sequential(
            InputEmbedding(d_model=embed_size, vocab_size=tgt_vocab_size),
            comp.SinusoidalPE(d_model=embed_size, max_len=tgt_max_length),
            nn.Dropout(0.1),
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
        self.device = device

    # def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
    #     # src: (N, src_len)
    #     src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    #     # src_mask: (N, 1, 1, src_len)
    #     return src_mask.to(self.device)

    # def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
    #     # tgt: (N, tgt_len)
    #     N, tgt_len = tgt.shape
    #     tgt_mask = torch.tril(
    #         torch.ones((tgt_len, tgt_len), device=self.device)
    #     ).expand(N, 1, tgt_len, tgt_len)
    #     # tgt_mask: (N, 1, tgt_len, tgt_len)
    #     return tgt_mask

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
        # src_mask = self.make_src_mask(src)
        # tgt_mask = self.make_tgt_mask(tgt)

        enc_src = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_src, tgt_mask, src_mask)
        output = self.project(dec_output)

        return output

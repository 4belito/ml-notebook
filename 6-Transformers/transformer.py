import torch
import torch.nn as nn


def attention_mask(seq_len, learned_embedding):
    # mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    embed = learned_embedding.new_full(
        (*learned_embedding.shape[:-1], seq_len, seq_len), float("-inf")
    )
    pos = torch.arange(seq_len)
    rel_pos = pos[:, None] - pos[None, :]
    valid_pos = (rel_pos >= 0) & (rel_pos < learned_embedding.shape[-1])
    embed[..., valid_pos] = learned_embedding[..., rel_pos[valid_pos]]
    return embed


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, res_pos_length=128):  #
        super().__init__()
        self.res_pos = nn.Parameter(torch.zeros(num_heads, res_pos_length))
        self.self_att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.in_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        # Self-attention block
        x_norm = self.in_norm(x)
        attn_mask = attention_mask(x.shape[1], self.res_pos)
        x = (
            x + self.self_att(x_norm, x_norm, x_norm, attn_mask=attn_mask)[0]
        )  # , attn_mask=attn_mask
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        self.network = nn.Sequential(
            nn.Embedding(128, embed_dim),
            *[TransformerLayer(embed_dim, num_heads) for _ in range(num_layers)],
            nn.Linear(embed_dim, 128),
        )

    def forward(self, x):
        return self.network(x)


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"{device=}")
    with open(__file__) as f:
        code = f.read()
    tokens = torch.as_tensor([ord(c) for c in code]).to(device)  # (seq_len, batch_size)
    net = Transformer(embed_dim=128, num_heads=8, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)

    for epoch in range(200):
        pred = net(tokens[None, :-1])[0]  # (1, seq_len, vocab_size)
        loss = nn.functional.cross_entropy(pred, tokens[1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 1 == 0:
            # print(f"target: {tokens[1:11]}")
            # print(f"pred : {pred[:10].argmax(-1)}")
            print(f"Epoch {epoch}: loss={loss.item():.4f}")


if __name__ == "__main__":
    # learned_embed = torch.arange(6).float().view(2, 3)
    # print(attention_mask(5, learned_embed))
    train()

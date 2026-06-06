import matplotlib.pyplot as plt
from torch import Tensor


def plot_embeddingdims(x: Tensor, emb: Tensor, title: str, n_col: int = 4):
    emb_dim = emb.shape[-1]
    _, ax = plt.subplots(
        (emb_dim - 1) // n_col + 1,
        n_col,
        figsize=(10, 2.5 * ((emb_dim - 1) // n_col + 1)),
        sharex=True,
    )
    for i in range(emb_dim):
        r, c = i // n_col, i % n_col
        ax[r, c].plot(x, emb[:, i].numpy())
        ax[r, c].set_title(f"dimension {i + 1}")
        if c == 0:
            ax[r, c].set_ylabel("Embedding value")
        if r == (emb_dim - 1) // n_col:
            ax[r, c].set_xlabel("x")

    plt.tight_layout()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_embedding(emb: Tensor, title: str, cmap: str = "RdBu", vmin: float = -1, vmax: float = 1):
    emb_plot = emb.numpy()
    plt.figure(figsize=(8, 4))
    plt.imshow(
        emb_plot,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="Embedding value")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Position")
    plt.title(title)
    plt.tight_layout()
    plt.show()

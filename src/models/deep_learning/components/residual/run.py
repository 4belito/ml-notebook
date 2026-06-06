import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

import models.deep_learning.components as mynn
from config import DATA_EXTERNAL
from helpers import get_device


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: nn.Module | None = None,
        res_connect: bool = False,
    ):
        super().__init__()  # type: ignore
        if activation is None:
            activation = nn.ReLU()

        layers: list[nn.Module] = [nn.Flatten()]
        in_dim = input_dim
        for h_dim in hidden_dims:
            block = nn.Sequential(
                nn.Linear(in_dim, h_dim), nn.LayerNorm(h_dim), activation
            )
            if res_connect:
                layers.append(mynn.ResidualConnection(block, in_dim, h_dim))
            else:
                layers.append(block)
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim, bias=False))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


device = get_device()
print(f"Using device: {device}")

size = (128, 128)
DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(size), torchvision.transforms.ToTensor()]
)
train_dataset = torchvision.datasets.Flowers102(
    root=DATA_EXTERNAL,
    split="train",
    download=True,
    transform=transform,
)
test_dataset = torchvision.datasets.Flowers102(
    root=DATA_EXTERNAL,
    split="test",
    download=True,
    transform=transform,
)


# %%
def visualize_image(img: torch.Tensor) -> Image.Image:
    return Image.fromarray((img.permute(1, 2, 0) * 255).to(torch.uint8).numpy())


visualize_image(train_dataset[1][0])

# %%
lr = 0.001
momentum = 0.9
batch_size = 32
loss = torch.nn.CrossEntropyLoss()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # type: ignore
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)  # type: ignore

# %% [markdown]
# ### Train

# %% [markdown]
# #### MLP

# %%
torch.manual_seed(1)  # type: ignore

mlp = MLP(
    input_dim=size[0] * size[1] * 3, hidden_dims=[512, 512, 512], output_dim=102
).to(device)


optimizer = torch.optim.SGD(mlp.parameters(), lr=lr, momentum=momentum)

mlp.train()
losses: list[float] = []
for epoch in range(30):
    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)
        pred_y = mlp(img)
        optimizer.zero_grad()
        loss_value = loss(pred_y, label)
        loss_value.backward()
        optimizer.step()  # type: ignore
        losses.append(loss_value.item())
    if epoch % 1 == 0:
        print(f"Epoch {epoch}, loss: {sum(losses) / len(losses)}")

# %%
torch.manual_seed(1)  # type: ignore

skipmlp = MLP(
    input_dim=size[0] * size[1] * 3,
    hidden_dims=[512, 512, 512],
    output_dim=102,
    res_connect=True,
).to(device)


optimizer = torch.optim.SGD(skipmlp.parameters(), lr=lr, momentum=momentum)

skipmlp.train()
skip_losses: list[float] = []
for epoch in range(30):
    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)
        pred_y = skipmlp(img)
        optimizer.zero_grad()
        loss_value = loss(pred_y, label)
        loss_value.backward()
        optimizer.step()  # type: ignore
        skip_losses.append(loss_value.item())
    if epoch % 1 == 0:
        print(f"Epoch {epoch}, loss: {sum(skip_losses) / len(skip_losses)}")

# %%
plt.plot(losses, linestyle="-", label="MLP")
plt.plot(skip_losses, linestyle="-", label="MLP & ResConn")
plt.title("Residual Connections Effect on Training Loss")
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# ### Evaluation

# %%
accuracy: float | None = None
for test_images, test_labels in test_loader:
    test_images, test_labels = test_images.to(device), test_labels.to(device)
    pred_test = mlp(test_images)
    accuracy = ((pred_test.argmax(dim=1) == test_labels).float().mean()).item()
print(f"Test accuracy: {accuracy}")

# %%
for test_images, test_labels in test_loader:
    test_images, test_labels = test_images.to(device), test_labels.to(device)
    prred_test = skipmlp(test_images)
    accuracy = ((prred_test.argmax(dim=1) == test_labels).float().mean()).item()
print(f"Test accuracy: {accuracy}")

import numpy as np
import torch
import torchvision
import torchvision.transforms as tform
from fire import Fire
from torch.utils.tensorboard import SummaryWriter

from model import ConvNet


def train():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    n_epochs = 100
    size = (128, 128)
    train_transform = tform.Compose(
        [
            tform.RandomResizedCrop(size, antialias=True),
            tform.RandomHorizontalFlip(),
            tform.ToTensor(),
            tform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = tform.Compose(
        [
            tform.Resize(size),
            tform.ToTensor(),
            tform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = torchvision.datasets.Flowers102(
        "./flowers", "train", transform=train_transform, download=True
    )
    val_dataset = torchvision.datasets.Flowers102(
        "./flowers", "val", transform=val_transform, download=True
    )
    test_dataset = torchvision.datasets.Flowers102(
        "./flowers", "test", transform=val_transform, download=True
    )

    writer = SummaryWriter()
    writer.add_graph(ConvNet(channels_l0=32, n_blocks=4), torch.zeros(1, 3, *size))
    writer.add_images("train_images", torch.stack([train_dataset[i][0] for i in range(32)]))
    # writer.flush()

    net = ConvNet(channels_l0=32, n_blocks=4).to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=0.005, weight_decay=1e-4)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=4
    )

    global_step = 0
    for epoch in range(n_epochs):
        net.train()
        train_accuracy = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            output = net(data)
            loss = torch.nn.functional.cross_entropy(output, label)

            train_accuracy.extend((output.argmax(dim=-1) == label).cpu().detach().float().numpy())

            optim.zero_grad()
            loss.backward()
            optim.step()

            # epoch_loss.append(loss.item())
            writer.add_scalar("train/loss", loss.item(), global_step=global_step)
            global_step += 1

        writer.add_scalar("train/accuracy", np.mean(train_accuracy), global_step=epoch)

        net.eval()
        val_accuracy = []
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)
            with torch.inference_mode():
                output = net(data)
            val_accuracy.extend((output.argmax(dim=-1) == label).cpu().detach().float().numpy())
        writer.add_scalar("val/accuracy", np.mean(val_accuracy), global_snstep=epoch)
        writer.flush()

        # Early stopping condition (optional)
        if epoch % 10 == 0:
            torch.save(net.state_dict(), f"model_{epoch}.pth")


if __name__ == "__main__":
    Fire(train)

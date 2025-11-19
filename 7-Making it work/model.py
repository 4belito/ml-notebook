import torch.nn as nn


class ConvNet(nn.Module):
    class Block(nn.Module):
        def __init__(self, in_channels, out_channels, stride, kernel_size=3):
            super().__init__()
            padding = (kernel_size - 1) // 2

            self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.n1 = nn.GroupNorm(1, out_channels)
            self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.n2 = nn.GroupNorm(1, out_channels)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()

            self.skip = (
                nn.Conv2d(in_channels, out_channels, 1, stride, 0)
                if in_channels != out_channels
                else nn.Identity()
            )

        def forward(self, x0):
            x = self.relu1(self.n1(self.c1(x0)))
            x = self.relu2(self.n2(self.c2(x)))
            return self.skip(x0) + x

    def __init__(self, channels_l0=64, n_blocks=4):
        super().__init__()
        cnn_layers = [nn.Conv2d(3, channels_l0, kernel_size=11, stride=2, padding=5), nn.ReLU()]
        c1 = channels_l0
        for i in range(n_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, stride=2))
            c1 = c2
        cnn_layers.append(nn.Conv2d(c1, 102, kernel_size=1))
        self.network = nn.Sequential(*cnn_layers)

    def forward(self, x):
        return self.network(x).mean(dim=-1).mean(dim=-1)

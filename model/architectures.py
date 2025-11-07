import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU()):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Flatten())
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation)
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SkipMLP(nn.Module):
    class SkipBlock(nn.Module):
        def __init__(self, in_dim, out_dim, activation=nn.ReLU()):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            if out_dim > 1:
                self.norm = nn.LayerNorm(out_dim)
            self.activation = activation
            if in_dim != out_dim:
                self.skip = nn.Linear(in_dim, out_dim)
            else:
                self.skip = nn.Identity()

        def forward(self, x):
            out = self.linear(x)
            out = self.norm(out)
            out = self.activation(out)
            return out + self.skip(x)

    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU()):
        super().__init__()
        layers = [nn.Flatten()]
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(self.SkipBlock(in_dim, h_dim, activation))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim, bias=False))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

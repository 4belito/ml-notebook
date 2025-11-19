import torch


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=torch.nn.ReLU()):
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(in_dim, h_dim))
            layers.append(activation)
            in_dim = h_dim
        layers.append(torch.nn.Linear(in_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

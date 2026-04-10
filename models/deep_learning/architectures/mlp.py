from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation_cls: type[nn.Module] = nn.ReLU,
        flatten: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        if flatten:
            layers.append(nn.Flatten())
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation_cls())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

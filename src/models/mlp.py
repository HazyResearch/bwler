import torch
import torch.nn as nn
from typing import List, Union, Optional

from .embeddings import make_embedding, infer_embedding_dim


class MLP(nn.Module):
    def __init__(
        self,
        n_dim: int = 1,
        n_layers: int = 3,
        hidden_dim: int = 32,
        activation = torch.nn.Tanh(),
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        embedding: str = "none",
        embedding_M: Optional[int] = None,
    ):
        """
        2-layer MLP that maps (B, n_dim) -> (B, 1)

        Args:
            hidden_dim: Dimension of hidden layer
            activation: Activation function (torch.nn module, default: Tanh)
        """
        super().__init__()
        self.n_dim = n_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.activation_fn = activation
        self.device = device
        self.dtype = dtype
        self.embedding_type = embedding

        # Create embedding layer
        self.embedding = make_embedding(
            kind=embedding, M=embedding_M, dtype=dtype, device=device
        )

        # Adjust input dimension based on embedding
        embedded_dim = self.embedding.output_dim

        # Build layers with embedding-adjusted input dimension
        layers = []
        layers.append(nn.Linear(embedded_dim, hidden_dim, dtype=dtype))
        layers.append(self.activation_fn)

        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=dtype))
            layers.append(self.activation_fn)

        layers.append(nn.Linear(hidden_dim, 1, dtype=dtype))

        self.network = nn.Sequential(*layers)
        self.to(device)

        print(f"MLP architecture: {self.network}")

    def make_grid(self, x: List[torch.Tensor]):
        # Form the meshgrid of points
        x_mesh = torch.meshgrid(*x, indexing="ij")
        x_mesh = torch.stack(x_mesh, dim=-1)
        return x_mesh

    def forward_grid(self, x_mesh: torch.Tensor):
        # Form the meshgrid of points
        out_shape = x_mesh.shape[:-1]
        x_mesh = x_mesh.reshape(-1, self.n_dim)
        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                x_mesh = self.fc[i](x_mesh)
            else:
                x_mesh = self.activation(self.fc[i](x_mesh))
        return x_mesh.reshape(out_shape)

    def forward_batch(self, x: torch.Tensor):
        """
        Forward pass of the network for a batch of points

        Args:
            x: Tensor of shape (B, n_dim)

        Returns:
            Tensor of shape (B, 1)
        """
        assert x.ndim == 2 and x.shape[1] == self.n_dim
        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                x = self.fc[i](x)
            else:
                x = self.activation(self.fc[i](x))
        return x

    def interpolate(self, x: List[torch.Tensor]):
        return self.forward(x)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x = inputs[0]  # Assuming single input tensor for 1D problems

        # Apply embedding
        x_embedded = self.embedding(x)

        return self.network(x_embedded).squeeze(-1)
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x = inputs[0]  # Assuming single input tensor for 1D problems

        # Apply embedding
        x_embedded = self.embedding(x)

        return self.network(x_embedded).squeeze(-1)

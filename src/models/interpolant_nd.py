import torch
import torch.nn as nn
from typing import List, Tuple, Callable, Union, Optional

from .utils.fornberg import D_banded_fornberg

EPS = 1e-14


class SpectralInterpolationND(nn.Module):
    def __init__(
        self,
        Ns: List[int],
        bases: List[str],
        domains: List[Tuple[float, float]],
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        fd_k: Optional[List[int]] = None,
    ):
        """
        ND interpolation using spectral methods

        Args:
            Ns: List of number of points per direction
            bases: List of bases per direction, either 'fourier' or 'chebyshev'
            domains: List of tuples of (min, max) per direction
            device: Device to use for computation
            dtype: Data type to use for computation
            fd_k: List of half-bandwidths for FD stencils (None means use spectral)
        """
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        # Store domain information
        assert len(Ns) == len(bases) == len(domains)
        self.n_dim = len(Ns)

        # Initialize FD parameters
        if fd_k is None:
            fd_k = [None] * self.n_dim

        assert len(fd_k) == self.n_dim
        self.fd_k = fd_k

        self.Ns = Ns
        self.bases = bases
        self.domains = domains
        self.domain_lengths = [domain[1] - domain[0] for domain in domains]

        # Set up nodes and weight/frequencies for interpolation (as needed)
        self.nodes = [None for _ in range(self.n_dim)]
        self.nodes_standard = [None for _ in range(self.n_dim)]
        self._from_standard = [None for _ in range(self.n_dim)]
        self._to_standard = [None for _ in range(self.n_dim)]
        self.cheb_weights = [None for _ in range(self.n_dim)]
        self.k = [None for _ in range(self.n_dim)]

        for dim in range(self.n_dim):
            if self.bases[dim] == "chebyshev":
                i = torch.linspace(
                    0, 1, self.Ns[dim], device=self.device, dtype=self.dtype
                )
                self.nodes_standard[dim] = torch.cos(torch.pi * i)
                # Compute barycentric weights for Chebyshev
                N = self.Ns[dim]
                weights = torch.ones(N, device=self.device, dtype=self.dtype)
                weights[0] *= 0.5
                weights[-1] *= 0.5
                weights[1::2] = -1
                self.cheb_weights[dim] = weights
                self.k[dim] = None
            elif self.bases[dim] == "fourier":
                self.nodes_standard[dim] = torch.linspace(
                    0,
                    2 * torch.pi,
                    self.Ns[dim] + 1,
                    device=self.device,
                    dtype=self.dtype,
                )[:-1]
                # Compute FFT frequencies
                self.k[dim] = torch.fft.fftfreq(self.Ns[dim]) * self.Ns[dim]
                self.k[dim] = self.k[dim].to(self.device, dtype=self.dtype)
            else:
                raise ValueError(f"Unknown basis: {self.bases[dim]}")

            # Set up domain mapping functions for this dimension
            domain = self.domains[dim]
            if self.bases[dim] == "chebyshev":
                self._to_standard[dim] = (
                    lambda x, d=dim: 2
                    * (x - self.domains[d][0])
                    / self.domain_lengths[d]
                    - 1
                )
                self._from_standard[dim] = (
                    lambda x, d=dim: self.domains[d][0]
                    + (x + 1) * self.domain_lengths[d] / 2
                )
            elif self.bases[dim] == "fourier":
                self._to_standard[dim] = (
                    lambda x, d=dim: 2
                    * torch.pi
                    * (x - self.domains[d][0])
                    / self.domain_lengths[d]
                )
                self._from_standard[dim] = lambda x, d=dim: self.domains[d][
                    0
                ] + self.domain_lengths[d] * x / (2 * torch.pi)
            else:
                raise ValueError(f"Unknown basis: {self.bases[dim]}")

            # Map standard nodes to physical domain
            self.nodes[dim] = self._from_standard[dim](self.nodes_standard[dim])

        # Set up diff matrices cache
        self._diff_matrices = [{} for _ in range(self.n_dim)]

        # Create mesh grid of nodes
        mesh_args = [self.nodes[d] for d in range(self.n_dim)]
        self.mesh = torch.meshgrid(*mesh_args, indexing="ij")

        # Learnable values at node points
        self.values = nn.Parameter(
            torch.zeros(self.Ns, device=self.device, dtype=self.dtype)
        )

    def load_values_from_model(self, model: nn.Module):
        with torch.no_grad():
            self.values.data = model(self.nodes)

    ############################################################
    # Compute derivative matrices
    ############################################################

    def _compute_cheb_derivative_matrix(
        self, nodes: torch.Tensor, domain_length: float
    ) -> torch.Tensor:
        """
        Compute the Chebyshev differentiation matrix for 1D Chebyshev-Gauss-Lobatto points.

        Args:
            nodes: Chebyshev-Gauss-Lobatto nodes
            domain_length: Physical domain length for scaling

        Returns:
            D: Differentiation matrix (N x N)
        """
        N = len(nodes)
        D = torch.zeros((N, N), dtype=nodes.dtype, device=nodes.device)

        # Compute weights for endpoints
        c = torch.ones(N, dtype=nodes.dtype, device=nodes.device)
        c[0] = 2
        c[-1] = 2

        # Compute off-diagonal entries
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = c[i] / c[j] * (-1) ** (i + j) / (nodes[i] - nodes[j])

        # Fill diagonal using negative sum trick
        D.diagonal().copy_(-torch.sum(D, dim=1))

        # Scale for domain transformation
        D = D * (2.0 / domain_length)

        return D

    def _compute_fourier_derivative_matrix(
        self, nodes: torch.Tensor, domain_length: float
    ) -> torch.Tensor:
        """
        Compute the Fourier differentiation matrix for 1D equispaced nodes.

        Args:
            nodes: Equispaced nodes from 0 to 2π
            domain_length: Physical domain length for scaling

        Returns:
            D: Differentiation matrix (N x N)
        """
        N = len(nodes)
        D = torch.zeros((N, N), dtype=nodes.dtype, device=nodes.device)

        # Create index matrices
        i, j = torch.meshgrid(
            torch.arange(N, dtype=nodes.dtype, device=nodes.device),
            torch.arange(N, dtype=nodes.dtype, device=nodes.device),
            indexing="ij",
        )

        # Compute off-diagonal elements using cotangent formula
        mask = i != j
        diff = (i[mask] - j[mask]) * (-1) ** (i[mask] - j[mask])
        D[mask] = 0.5 * torch.tan(torch.pi * diff / N).reciprocal()

        # Diagonal elements are 0 for periodic functions
        D.diagonal().zero_()

        # Scale for domain transformation
        D = D * (2 * torch.pi / domain_length)

        return D

    def _compute_fornberg_derivative_matrix(
        self, nodes: torch.Tensor, domain_length: float, k: int, m: int = 1
    ) -> torch.Tensor:
        """
        Compute the Fornberg differentiation matrix for 1D nodes.

        Args:
            nodes: Physical domain nodes
            domain_length: Physical domain length for scaling
            k: Half-bandwidth of the stencil (2k+1 points)
            m: Derivative order (default: 1)

        Returns:
            D: Differentiation matrix (N x N)
        """

        assert (
            m + 1
        ) // 2 <= k, (
            f"For {m}th derivative, need stencil size k >= {(m + 1) // 2}, got k={k}"
        )
        D = D_banded_fornberg(nodes, k=k, m=m)
        return D

    def derivative_matrix(self, k: Tuple[int, ...]) -> torch.Tensor:
        """
        Get mixed derivative matrix D^k where k is a tuple of derivative orders

        Args:
            k: List/tuple of length n_dim specifying derivative order in each dimension
               e.g., (2,0,1) means second derivative in x, none in y, first in z

        Returns:
            Matrix operator for the mixed derivative
        """
        assert (
            len(k) == self.n_dim
        ), f"Expected {self.n_dim} derivative orders, got {len(k)}"

        # Get 1D matrices for each dimension
        matrices = []
        for dim in range(self.n_dim):
            if k[dim] == 0:
                # Identity matrix for this dimension
                matrices.append(
                    torch.eye(
                        self.Ns[dim],
                        dtype=self.nodes_standard[dim].dtype,
                        device=self.nodes_standard[dim].device,
                    )
                )
            else:
                # Compute/get cached derivative matrix
                if k[dim] not in self._diff_matrices[dim]:
                    if 1 not in self._diff_matrices[dim]:
                        if self.bases[dim] == "chebyshev":
                            D = self._compute_cheb_derivative_matrix(
                                nodes=self.nodes_standard[dim],
                                domain_length=self.domain_lengths[dim],
                            )
                        elif self.bases[dim] == "fourier":
                            D = self._compute_fourier_derivative_matrix(
                                nodes=self.nodes_standard[dim],
                                domain_length=self.domain_lengths[dim],
                            )
                        else:
                            raise ValueError(f"Unknown basis: {self.bases[dim]}")
                        self._diff_matrices[dim][1] = D

                    # Compose for higher derivatives
                    Dk = self._diff_matrices[dim][1]
                    for _ in range(k[dim] - 1):
                        Dk = Dk @ self._diff_matrices[dim][1]
                    self._diff_matrices[dim][k[dim]] = Dk

                matrices.append(self._diff_matrices[dim][k[dim]])

        # Compute Kronecker product
        D = matrices[0]
        for dim in range(1, self.n_dim):
            D = torch.kron(D, matrices[dim])

        return D

    ############################################################
    # Interpolation
    ############################################################

    def _cheb_interpolate_1d(
        self,
        x_eval: torch.Tensor,
        values: torch.Tensor,
        nodes_std: torch.Tensor,
        to_std: Callable,
        weights: torch.Tensor,
        eps: float = EPS,
    ):
        """
        Helper for 1D Chebyshev interpolation along last axis.
        Uses the second barycentric formula:
            f(x) = sum_{j=0}^{N-1} f_j * w_j / (x - x_j) / sum_{j=0}^{N-1} w_j / (x - x_j)

        Args:
            x_eval: shape (B1, B) - points to evaluate at
            values: shape (B2, B, N) - function values at nodes
            nodes_std: shape (N,) - standard Chebyshev nodes
            to_std: function - maps from physical to standard domain
            weights: shape (N,) - barycentric weights

        Returns:
            shape (B1, B2, B) - interpolated values
        """
        x_eval_standard = to_std(x_eval)  # (B1, B)

        # Reshape inputs for broadcasting:
        # x_eval: (B1, 1, B, 1)
        # values: (1, B2, B, N)
        # nodes: (1, 1, 1, N)
        # weights: (1, 1, 1, N)

        x_eval_expanded = x_eval_standard.unsqueeze(1).unsqueeze(-1)  # (B1, 1, B, 1)
        values_expanded = values.unsqueeze(0)  # (1, B2, B, N)
        nodes_expanded = nodes_std.reshape(1, 1, 1, -1)
        weights_expanded = weights.reshape(1, 1, 1, -1)

        # Compute distances - result is (B1, B2, B, N)
        d_x = x_eval_expanded - nodes_expanded

        small_diff = torch.abs(d_x) < eps
        small_diff_max = torch.max(small_diff, dim=-1, keepdim=True).values

        d_x = torch.where(small_diff_max, torch.zeros_like(d_x), 1.0 / d_x)
        d_x[small_diff] = 1

        # Compute weighted sum along last axis
        f_eval_num = torch.sum(
            values_expanded * d_x * weights_expanded, dim=-1
        )  # (B1, B2, B)
        f_eval_denom = torch.sum(d_x * weights_expanded, dim=-1)  # (B1, B2, B)

        return f_eval_num / f_eval_denom

    def _cheb_interpolate_1ofnd(
        self,
        values: torch.Tensor,
        x_eval: torch.Tensor,
        dim: int,
        nodes_std: torch.Tensor,
        to_std: Callable,
        weights: torch.Tensor,
        eps: float = EPS,
    ):
        """
        Interpolate along a specific Chebyshev dimension of a tensor.

        Args:
            values: Tensor of shape (..., N, ...), where N is the size of the Chebyshev dim.
            x_eval: Tensor of shape (m,), points to evaluate along the Chebyshev dimension.
            dim: Integer, the axis corresponding to the Chebyshev dimension in `values`.
            nodes_std: Tensor of shape (N,), the Chebyshev nodes.
            to_std: Function mapping physical to standard domain.
            weights: Tensor of shape (N,), the barycentric weights.
            eps: Small value to handle division by zero.

        Returns:
            Tensor of shape (..., m, ...), with the Chebyshev dimension replaced by interpolated values.
        """

        # Step 1: Move the Chebyshev axis to the last position for simplicity
        values_moved = values.movedim(dim, -1)  # (..., N)
        batch_shape, N = values_moved.shape[:-1], values_moved.shape[-1]
        m = x_eval.shape[0]  # Number of evaluation points

        # Step 2: Reshape for batch broadcasting
        # - Add a singleton dimension to `values` for x_eval
        # - Add a singleton dimension to `x_eval` for values
        values_reshaped = values_moved.reshape(-1, 1, N)  # (..., None, N)
        x_eval_reshaped = x_eval[:, None]  # (m, 1)

        # Step 3: Call the 1D Chebyshev interpolation helper
        interpolated = self._cheb_interpolate_1d(
            x_eval=x_eval_reshaped,  # (m, 1)
            values=values_reshaped,  # (..., None, N)
            nodes_std=nodes_std,  # (N,)
            to_std=to_std,  # Function
            weights=weights,  # (N,)
            eps=eps,
        )  # (..., m, 1)

        # Step 4: Restore the original dimension layout
        interpolated = interpolated.reshape(m, *batch_shape).movedim(
            0, dim
        )  # (..., m, ...)

        return interpolated

    def _fourier_interpolate_1d(
        self,
        x_eval: torch.Tensor,
        values: torch.Tensor,
        to_std: Callable,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Helper for 1D Fourier interpolation along last axis

        Args:
            x_eval: shape (B1, B) - points to evaluate at
            values: shape (B2, B, N) - function values at nodes
            to_std: function - maps from physical to standard domain
            k: shape (N,) - frequency modes

        Returns:
            shape (B1, B2, B) - interpolated values
        """
        N = values.shape[-1]
        x_eval_standard = to_std(x_eval)  # (B1, B)

        # Compute FFT along last axis
        coeffs = torch.fft.fft(values, dim=-1)  # (B2, B, N)

        # Reshape inputs for broadcasting:
        # x_eval: (B1, 1, B, 1)
        # coeffs: (1, B2, B, N)
        # k: (1, 1, 1, N)
        x_eval_expanded = x_eval_standard.unsqueeze(1).unsqueeze(-1)  # (B1, 1, B, 1)
        coeffs_expanded = coeffs.unsqueeze(0)  # (1, B2, B, N)
        k_expanded = k.reshape(1, 1, 1, -1)

        # Compute Fourier matrix - result is (B1, 1, B, N)
        x_matrix = x_eval_expanded * k_expanded
        fourier_matrix = torch.exp(1j * x_matrix)

        # Matrix multiply and sum along last axis - result is (B1, B2, B)
        result = torch.sum(fourier_matrix * coeffs_expanded, dim=-1)
        return torch.real(result) / N

    def _fourier_interpolate_1ofnd(
        self,
        values: torch.Tensor,
        x_eval: torch.Tensor,
        dim: int,
        to_std: Callable,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Interpolate along a specific Fourier dimension of a tensor.

        Args:
            values: Tensor of shape (..., N, ...), where N is the size of the Fourier dim.
            x_eval: Tensor of shape (m,), points to evaluate along the Fourier dimension.
            dim: Integer, the axis corresponding to the Fourier dimension in `values`.
            to_std: Function mapping physical to standard domain.
            k: Tensor of shape (N,), the Fourier frequencies.

        Returns:
            Tensor of shape (..., m, ...), with the Fourier dimension replaced by interpolated values.
        """

        # Step 1: Move the Fourier axis to the last position for simplicity
        values_moved = values.movedim(dim, -1)
        batch_shape, N = values_moved.shape[:-1], values_moved.shape[-1]
        m = x_eval.shape[0]  # Number of evaluation points

        # Step 2: Reshape for batch broadcasting
        # - Add a singleton dimension to `values` for x_eval
        # - Add a singleton dimension to `x_eval` for values
        values_reshaped = values_moved.reshape(-1, 1, N)
        x_eval_reshaped = x_eval[:, None]

        # Step 3: Call the 1D Fourier interpolation helper
        interpolated = self._fourier_interpolate_1d(
            x_eval=x_eval_reshaped,  # Shape (m, 1)
            values=values_reshaped,  # Shape (..., None, N)
            to_std=to_std,  # Function
            k=k,  # Shape (..., N, 1)
        )

        # Step 4: Restore the original dimension layout
        interpolated = interpolated.reshape(m, *batch_shape).movedim(0, dim)

        return interpolated

    def interpolate(self, x_eval: List[torch.Tensor], values=None) -> torch.Tensor:
        """
        Interpolate the function at the given points

        Args:
            x_eval: List of tensors of shapes (m1,), (m2,), ..., (m_ndim,) - points to evaluate at
            values: Tensor of shape (N1, N2, ..., N_ndim) - function values at nodes. Defaults to self.values.

        Returns:
            Tensor of shape (m1, m2, ..., m_ndim) - interpolated values
        """
        if values is not None:
            assert values.shape == self.values.shape
            interpolated = values
        else:
            interpolated = self.values
        for dim in range(self.n_dim):
            if self.bases[dim] == "chebyshev":
                interpolated = self._cheb_interpolate_1ofnd(
                    values=interpolated,
                    x_eval=x_eval[dim],
                    dim=dim,
                    nodes_std=self.nodes_standard[dim],
                    to_std=self._to_standard[dim],
                    weights=self.cheb_weights[dim],
                )
            elif self.bases[dim] == "fourier":
                interpolated = self._fourier_interpolate_1ofnd(
                    values=interpolated,
                    x_eval=x_eval[dim],
                    dim=dim,
                    to_std=self._to_standard[dim],
                    k=self.k[dim],
                )
            else:
                raise ValueError(f"Unknown basis: {self.bases[dim]}")

        return interpolated

    ############################################################################################
    # Batch interpolation
    ############################################################################################

    def _compute_cheb_basis_1d_batch(
        self,
        x_eval: torch.Tensor,  # shape (B,)
        nodes_std: torch.Tensor,  # shape (N,)
        to_std: Callable,
        weights: torch.Tensor,  # shape (N,)
        eps: float = EPS,
    ) -> torch.Tensor:
        """
        Compute the Chebyshev basis functions L_j(x_eval[b]) for all b in x_eval.
        Uses the second barycentric formula:
            L_j(x) = w_j / (x - x_j) / sum_{i=0}^{N-1} w_i / (x - x_i)

        Args:
            x_eval: shape (B,) - points to evaluate at
            nodes_std: shape (N,) - standard Chebyshev nodes
            to_std: function - maps from physical to standard domain
            weights: shape (N,) - barycentric weights

        Returns:
            shape (B, N) - Chebyshev basis functions evaluated at x_eval
        """
        x_eval_standard = to_std(x_eval)

        # Reshape inputs for broadcasting:
        # x_eval: (B, 1)
        # nodes_std: (1, N)
        # weights: (1, N)
        x_eval_expanded = x_eval_standard.unsqueeze(1)  # (B, 1)
        nodes_std_expanded = nodes_std.reshape(1, -1)  # (1, N)
        weights_expanded = weights.reshape(1, -1)  # (1, N)

        # Compute distances - result is (B, N)
        d_x = x_eval_expanded - nodes_std_expanded

        small_diff = torch.abs(d_x) < eps
        small_diff_max = torch.max(small_diff, dim=-1, keepdim=True).values

        d_x = torch.where(small_diff_max, torch.zeros_like(d_x), 1.0 / d_x)
        d_x[small_diff] = 1

        # Compute weighted sum along last axis
        f_eval_num = d_x * weights_expanded  # (B, N)
        f_eval_denom = torch.sum(d_x * weights_expanded, dim=-1)  # (B,)

        return f_eval_num / f_eval_denom.unsqueeze(-1)  # (B, N)

    def _compute_fourier_basis_1d_batch(
        self,
        x_eval: torch.Tensor,  # shape (B,)
        to_std: Callable,
        k: torch.Tensor,  # shape (N,)
    ) -> torch.Tensor:
        """
        Compute the Fourier basis functions exp(1j * k * x_eval[b]) for all b in x_eval.

        Args:
            x_eval: shape (B,) - points to evaluate at
            to_std: function - maps from physical to standard domain
            k: shape (N,) - frequency modes

        Returns:
            shape (B, N) - Fourier basis functions evaluated at x_eval
        """
        x_eval_standard = to_std(x_eval)
        phase = x_eval_standard.unsqueeze(1) * k.reshape(
            1, -1
        )  # (B, 1) * (1, N) -> (B, N)
        complex_dtype = (
            torch.complex128 if phase.dtype == torch.float64 else torch.complex64
        )
        return torch.exp(1j * phase.to(complex_dtype))  # (B, N)

    def interpolate_batch(
        self, x_eval: torch.Tensor, values: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Interpolate the function at the given points in batch.
        Uses the formula:
            f(x) = sum_{i_1=0}^{N_1-1} ... sum_{i_D=0}^{N_D-1} f_{i_1, ..., i_D} * prod_{d=1}^{D} L_{i_d}(x_eval[b, d])
            where L_{i_d}(x) is the dth Chebyshev basis function for the ith node.

        Args:
            x_eval: shape (B, D) - points to evaluate at
            values: Tensor of shape (N1, N2, ..., N_ndim) - function values at nodes. Defaults to self.values.

        Returns:
            shape (B,) - interpolated values
        """

        # Check that x_eval is a 2D tensor
        if x_eval.dim() != 2:
            raise ValueError(f"Expected 2D tensor (B, D), got shape {x_eval.shape}")
        # Check that x_eval has the correct number of dimensions
        if x_eval.shape[1] != self.n_dim:
            raise ValueError(f"Expected {self.n_dim} dimensions, got {x_eval.shape[1]}")

        # Check that values is the correct shape
        if values is not None:
            assert values.shape == self.values.shape
        else:
            values = self.values

        # Compute FFTN (if needed)
        fft_dims = [dim for dim, basis in enumerate(self.bases) if basis == "fourier"]

        # Compute the FFTN of the values
        if len(fft_dims) > 0:
            complex_dtype = (
                torch.complex128 if values.dtype == torch.float64 else torch.complex64
            )
            representation = torch.fft.fftn(
                values.to(complex_dtype), dim=tuple(fft_dims)
            )
            representation_is_complex = True
            fourier_norm_factor = 1.0 / torch.prod(
                torch.tensor([self.Ns[dim] for dim in fft_dims], dtype=torch.float64)
            )
        else:
            representation = values
            representation_is_complex = False

        # Compute basis matrices (Chebyshev or Fourier) and einsum indices
        basis_matrices = []
        einsum_basis_indices = []
        einsum_value_indices = []
        for dim in range(self.n_dim):

            # Create the einsum indices: i, j, k, ... -> i, j, k, ...
            basis_char = chr(ord("i") + dim)
            einsum_basis_indices.append(f"b{basis_char}")
            einsum_value_indices.append(basis_char)

            if self.bases[dim] == "chebyshev":
                # Compute Chebyshev basis functions
                basis_matrix = self._compute_cheb_basis_1d_batch(
                    x_eval=x_eval[:, dim],
                    nodes_std=self.nodes_standard[dim],
                    to_std=self._to_standard[dim],
                    weights=self.cheb_weights[dim],
                )
            elif self.bases[dim] == "fourier":
                # Compute Fourier basis functions
                basis_matrix = self._compute_fourier_basis_1d_batch(
                    x_eval=x_eval[:, dim], to_std=self._to_standard[dim], k=self.k[dim]
                )
            basis_matrices.append(basis_matrix)

            # If the representation is complex, convert the basis matrices to complex
            if representation_is_complex:
                for i in range(len(basis_matrices)):
                    basis_matrices[i] = basis_matrices[i].to(complex_dtype)
            else:
                for i in range(len(basis_matrices)):
                    assert basis_matrices[i].dtype == values.dtype

        # Create the einsum expression
        einsum_expr = (
            f"{','.join(einsum_basis_indices)},{''.join(einsum_value_indices)} -> b"
        )

        # Compute the interpolant
        interpolant = torch.einsum(einsum_expr, *basis_matrices, representation)

        # Normalize the interpolant by the FFT norm factor (if needed)
        if representation_is_complex:
            interpolant *= fourier_norm_factor
            interpolant = interpolant.real

        return interpolant  # shape (B,)

    def forward(self, x_eval: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the interpolant at arbitrary evaluation points

        Args:
            x_eval: Either:
                - List of tensors of shapes (m1,), (m2,), ..., (m_ndim,) - points to evaluate at. We perform interpolation along each dimension at these points.
                - Tensor of shape (B, D) where B is batch size and D is number of dimensions. We convert into a list of D tensors of shape (B,) and perform interpolation along each dimension at these points.

        Returns:
            Tensor of shape (m1, m2, ..., m_ndim) or (B,) depending on input format
        """
        if isinstance(x_eval, list) or isinstance(x_eval, tuple):
            return self.interpolate(x_eval, values=self.values)
        elif isinstance(x_eval, torch.Tensor):
            return self.interpolate_batch(x_eval, values=self.values)
        else:
            raise TypeError(f"Expected list of tensors or tensor, got {type(x_eval)}")

    ############################################################
    # Efficient derivatives
    ############################################################

    def _chebfft_derivative_1d(self, values: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Compute derivative along a single Chebyshev dimension using FFT method.
        Vectorized implementation for batch processing.

        Args:
            values: Tensor containing function values at Chebyshev points
            dim: The dimension along which to take the derivative
        """
        # Get size information
        N = self.Ns[dim] - 1

        # Move the target dimension to the end for processing
        values_moved = values.movedim(dim, -1)
        orig_shape = values_moved.shape

        # Reshape to 2D for batch processing (batch_size, N+1)
        values_flat = values_moved.reshape(-1, N + 1)
        batch_size = values_flat.shape[0]

        # Step 1: Extend data to length 2N using symmetry
        # Create a tensor of shape (batch_size, 2N)
        V = torch.zeros(batch_size, 2 * N, device=values.device)
        V[:, : N + 1] = values_flat
        V[:, N + 1 :] = torch.flip(values_flat[:, 1:N], [1])

        # Step 2: Calculate FFT
        Vhat = torch.fft.fft(V, dim=1)  # Shape: (batch_size, 2N)
        k = torch.arange(2 * N, device=values.device)
        k = torch.where(k <= N, k, k - 2 * N)

        # Step 3: Define Whom = ik*vhat, except Whom[N] = 0
        Whom = 1j * k * Vhat  # Broadcasting handles batch dimension
        Whom[:, N] = 0

        # Step 4: Inverse FFT to get derivative on equispaced grid
        W = torch.fft.ifft(Whom, dim=1).real  # Shape: (batch_size, 2N)

        # Step 5: Extract values and adjust for chain rule
        x = self.nodes_standard[dim]
        deriv = torch.zeros(batch_size, N + 1, device=values.device)

        # Interior points
        deriv[:, 1:N] = -W[:, 1:N] / torch.sqrt(1 - x[1:N] ** 2)

        # Endpoints using special formulas
        n = torch.arange(N + 1, device=values.device, dtype=torch.float64)
        n2 = n * n
        n2[0] *= 0.5  # Halve the n=0 term
        n2[N] *= 0.5  # Halve the n=N term

        # Get Chebyshev coefficients for all batches at once
        v_coeffs = Vhat[:, : N + 1] * (2.0 / N * torch.pi)  # Shape: (batch_size, N+1)

        # w0 = (1/2π)Σ' n²v̂ₙ
        deriv[:, 0] = torch.sum(n2 * v_coeffs.real, dim=1) / (2.0 * torch.pi)

        # wN = (1/2π)Σ' (-1)^(n+1) n²v̂ₙ
        signs = (-1) ** (n + 1)
        deriv[:, N] = torch.sum(signs * n2 * v_coeffs.real, dim=1) / (2.0 * torch.pi)

        # Restore original shape and dimension order
        deriv_reshaped = deriv.reshape(orig_shape)
        deriv_final = deriv_reshaped.movedim(-1, dim)

        # Scale for domain transformation if not on [-1, 1]
        domain_length = self.domain_lengths[dim]
        if domain_length != 2:  # if domain is not [-1, 1]
            deriv_final *= 2.0 / domain_length

        return deriv_final

    def _derivative_interpolant(
        self,
        k: Tuple[int, ...],
        use_spectral: bool = False,
    ) -> torch.Tensor:
        """
        Compute mixed derivative of interpolant using either spectral or FD methods.

        Args:
            k: Tuple of length n_dim specifying derivative order in each dimension
        """
        # Handle the case where k is a single integer
        if isinstance(k, int):
            k = (k,) + (0,) * (self.n_dim - 1)

        assert (
            len(k) == self.n_dim
        ), f"Expected {self.n_dim} derivative orders, got {len(k)}"

        # If all derivatives are zero, return values
        if all(ki == 0 for ki in k):
            return self.values

        result = self.values

        # Apply derivatives dimension by dimension
        for dim in range(self.n_dim):
            if k[dim] > 0:
                # Check if we should use FD for this dimension
                if not use_spectral and self.fd_k[dim] is not None:
                    # Use Fornberg FD method
                    # For higher derivatives, we need k >= ceil(m/2)
                    if self.fd_k[dim] < (k[dim] + 1) // 2:
                        raise ValueError(
                            f"For {k[dim]}th derivative, need stencil size k >= {(k[dim] + 1) // 2}, got k={self.fd_k[dim]}"
                        )

                    # Compute/get cached derivative matrix
                    if k[dim] not in self._diff_matrices[dim]:
                        D = self._compute_fornberg_derivative_matrix(
                            nodes=self.nodes[dim],
                            domain_length=self.domain_lengths[dim],
                            k=self.fd_k[dim],
                            m=k[dim],
                        )
                        self._diff_matrices[dim][k[dim]] = D
                    else:
                        D = self._diff_matrices[dim][k[dim]]

                    # Apply to the appropriate dimension
                    result_dot = torch.tensordot(
                        D,
                        result,
                        dims=([1], [dim]),
                    )
                    # Permute to [0, ..., ndims]
                    perm = list(range(len(result.shape)))
                    perm.pop(dim)
                    perm.insert(0, dim)
                    result = result_dot.permute(perm)
                else:
                    # Use spectral method
                    if self.bases[dim] == "chebyshev":
                        # Use FFT method for Chebyshev bases
                        for _ in range(k[dim]):
                            result = self._chebfft_derivative_1d(result, dim)
                    else:
                        # Use matrix method for Fourier
                        if 1 not in self._diff_matrices[dim]:
                            D = self._compute_fourier_derivative_matrix(
                                nodes=self.nodes_standard[dim],
                                domain_length=self.domain_lengths[dim],
                            )
                            self._diff_matrices[dim][1] = D
                        else:
                            D = self._diff_matrices[dim][1]
                        # Compose for higher derivatives
                        for _ in range(k[dim] - 1):
                            D = D @ self._diff_matrices[dim][1]

                        # Apply to the appropriate dimension
                        result_dot = torch.tensordot(
                            D,
                            result,
                            dims=([1], [dim]),
                        )
                        # Permute to [0, ..., ndims]
                        perm = list(range(len(result.shape)))
                        perm.pop(dim)
                        perm.insert(0, dim)
                        result = result_dot.permute(perm)
        return result

    def update_fd_params(self, fd_k: Optional[List[int]] = None):
        """
        Update the FD parameters during training.

        Args:
            fd_k: List of half-bandwidths for FD stencils (None means use spectral)
        """
        if fd_k is not None:
            assert len(fd_k) == self.n_dim
            self.fd_k = fd_k
            # Clear derivative matrix cache when stencil size changes
            self._diff_matrices = [{} for _ in range(self.n_dim)]

    def derivative(
        self,
        x_eval: Union[List[torch.Tensor], torch.Tensor],
        k: Tuple[int, ...],
        use_spectral: bool = False,
    ) -> torch.Tensor:
        """
        Compute mixed derivative of interpolant at arbitrary evaluation points

        Args:
            x_eval: Either:
                - List of tensors of shapes (m1,), (m2,), ..., (m_ndim,) - points to evaluate at. We perform interpolation along each dimension at these points.
                - Tensor of shape (B, D) where B is batch size and D is number of dimensions. We convert into a list of D tensors of shape (B,) and perform interpolation along each dimension at these points.
            k: Tuple of length n_dim specifying derivative order in each dimension
               e.g., (2,0,1) means second derivative in x, none in y, first in z

        Returns:
            Tensor of shape (m1, m2, ..., m_ndim) or (B,) depending on input format
        """
        # Compute derivative at nodes
        dk_nodes = self._derivative_interpolant(k, use_spectral=use_spectral)

        # Interpolate to evaluation points
        if isinstance(x_eval, list) or isinstance(x_eval, tuple):
            return self.interpolate(x_eval, values=dk_nodes)
        elif isinstance(x_eval, torch.Tensor):
            return self.interpolate_batch(x_eval, values=dk_nodes)
        else:
            raise TypeError(f"Expected list of tensors or tensor, got {type(x_eval)}")

    def _fd_derivative_at_nodes(
        self,
        dim: int,
        h: float,
        scheme: str = "central",
    ) -> torch.Tensor:
        """
        Central-difference FD derivative on the *physical* nodes.
        Boundaries are handled using spectral derivatives.

        Args:
            dim: int
                Spatial dimension to differentiate (0-based).
            h: float
                Physical step size along that dimension.
            scheme: str
                FD scheme to use (currently only 'central' is supported)
        """
        if scheme.lower() != "central":
            raise ValueError("Only central difference scheme is currently supported")

        N = self.Ns[dim]
        # Get spectral derivative for boundaries
        _spectral_deriv_tuple = [0 for _ in range(self.n_dim)]
        _spectral_deriv_tuple[dim] = 1
        deriv = self.derivative(
            self.nodes, _spectral_deriv_tuple, use_spectral=True
        )  # shape (N,)
        if N < 3:
            return deriv

        # Build shifted node lists for central difference
        nodes_p = list(self.nodes)
        nodes_m = list(self.nodes)
        nodes_p[dim] = self.nodes[dim] + h
        nodes_m[dim] = self.nodes[dim] - h

        # Interpolate at shifted points
        f_p = self.interpolate(nodes_p)
        f_m = self.interpolate(nodes_m)

        # Apply central difference to interior points
        slc = [slice(None)] * self.n_dim
        slc[dim] = slice(1, N - 1)
        interior = tuple(slc)

        deriv[interior] = (f_p[interior] - f_m[interior]) / (2.0 * h)

        return deriv

    def fd_derivative(
        self,
        x_eval: Union[List[torch.Tensor], torch.Tensor],
        dim: int,
        h: float,
        scheme: str = "central",
    ) -> torch.Tensor:
        """
        Compute FD derivative at arbitrary points by:
        1. computing nodal derivatives via central FD with spectral boundaries,
        2. re-interpolating that nodal derivative field.

        Args:
            x_eval: Points to evaluate at
            dim: Spatial dimension to differentiate
            h: Step size in physical units
            scheme: FD scheme to use (currently only 'central' is supported)
        """
        dk_nodes = self._fd_derivative_at_nodes(dim, h, scheme)
        if isinstance(x_eval, (list, tuple)):
            return self.interpolate(x_eval, values=dk_nodes)
        elif isinstance(x_eval, torch.Tensor):
            return self.interpolate_batch(x_eval, values=dk_nodes)
        else:
            raise TypeError(f"Expected list of tensors or tensor, got {type(x_eval)}")


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    torch.set_default_dtype(torch.float64)

    """
    Test the interpolant and its derivatives in 1D.
    We compare the tensor interpolation, node interpolation, batch interpolation,
    spectral derivatives, and finite-difference derivatives on f = sin(2πx).
    """

    print(f"========== Testing 1D interpolant and derivatives ==========")

    # Test function and its derivatives
    f_1d = lambda x: torch.sin(2 * torch.pi * x)
    f_1d_x = lambda x: 2 * torch.pi * torch.cos(2 * torch.pi * x)
    f_1d_xx = lambda x: -4 * torch.pi**2 * torch.sin(2 * torch.pi * x)

    # Test parameters
    N = 10  # Number of nodes
    N_dense = 100  # Number of evaluation points
    k_values = [1, 2, 3, 4, 5]  # FD stencil sizes to test

    # Create plots directory
    plot_dir = os.path.join(os.path.dirname(__file__), "../../plots/dev/interpolant_nd")
    os.makedirs(plot_dir, exist_ok=True)

    # Test both Chebyshev and Fourier bases
    for basis in ["chebyshev", "fourier"]:
        print(f"\nTesting {basis} basis")

        # Initialize interpolant
        interpolant = SpectralInterpolationND(
            Ns=[N + 1 if basis == "chebyshev" else N],
            bases=[basis],
            domains=[(0, 1)],
        )

        # Set values at nodes
        f_vals = f_1d(interpolant.nodes[0])
        interpolant.values.data = f_vals.reshape(interpolant.values.shape)

        # Create dense evaluation points
        x_eval = torch.linspace(0, 1, N_dense)
        f_eval = f_1d(x_eval)
        f_eval_x = f_1d_x(x_eval)
        f_eval_xx = f_1d_xx(x_eval)

        # Test interpolation
        interpolated = interpolant([x_eval])
        interp_error = torch.abs(interpolated - f_eval)

        # Test spectral derivatives
        spectral_dx = interpolant.derivative([x_eval], (1,))
        spectral_dx_error = torch.abs(spectral_dx - f_eval_x)

        spectral_dxx = interpolant.derivative([x_eval], (2,))
        spectral_dxx_error = torch.abs(spectral_dxx - f_eval_xx)

        # Test FD derivatives with different k values
        fd_errors = []
        for k in k_values:
            interpolant.update_fd_params(fd_k=[k])
            fd_dx = interpolant.derivative([x_eval], (1,), use_spectral=False)
            fd_dx_error = torch.abs(fd_dx - f_eval_x)
            fd_errors.append(fd_dx_error)

        # Create plots
        plt.figure(figsize=(15, 10))

        # Plot 1: Function and interpolation
        plt.subplot(2, 2, 1)
        plt.plot(x_eval, f_eval.detach().cpu().numpy(), "k-", label="Exact")
        plt.plot(
            x_eval, interpolated.detach().cpu().numpy(), "r--", label="Interpolated"
        )
        plt.scatter(interpolant.nodes[0], f_vals, c="k", label="Nodes")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(f"{basis} Interpolation")
        plt.legend()

        print(f"Interpolation error: {torch.mean(interp_error).item()}")

        # Plot 2: Interpolation error
        plt.subplot(2, 2, 2)
        plt.semilogy(
            x_eval, interp_error.detach().cpu().numpy(), "b-", label="Interpolation"
        )
        plt.semilogy(
            x_eval, spectral_dx_error.detach().cpu().numpy(), "g-", label="Spectral dx"
        )
        plt.semilogy(
            x_eval,
            spectral_dxx_error.detach().cpu().numpy(),
            "r-",
            label="Spectral dxx",
        )
        plt.xlabel("x")
        plt.ylabel("Absolute Error")
        plt.title("Spectral Error")
        plt.legend()

        print(f"Spectral error: {torch.mean(spectral_dx_error).item()}")

        # Plot 3: FD errors for different k
        plt.subplot(2, 2, 3)
        for i, k in enumerate(k_values):
            plt.semilogy(x_eval, fd_errors[i].detach().cpu().numpy(), label=f"k = {k}")
        plt.xlabel("x")
        plt.ylabel("Absolute Error")
        plt.title("FD Error")
        plt.legend()

        print(
            f"FD errors: {[torch.mean(fd_errors[i]).item() for i in range(len(fd_errors))]}"
        )

        # Plot 4: Error convergence
        plt.subplot(2, 2, 4)
        max_errors = [torch.max(err).item() for err in fd_errors]
        plt.semilogy(k_values, max_errors, "o-", label="FD")
        plt.xlabel("Stencil size k")
        plt.ylabel("Maximum Error")
        plt.title("Error vs Stencil Size")
        plt.legend()
        plt.grid(True, which="both", ls="-")

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{basis}_analysis.png"))
        plt.close()

    """
    Test the interpolant and its derivatives in 2D.
    We compare the tensor interpolation, node interpolation, batch interpolation,
    spectral derivatives, and finite-difference derivatives on f = sin(2πx)cos(2πy).
    """
    print(f"========== Testing 2D interpolant and derivatives ==========")

    # Initialize a 2D Chebyshev interpolant and evaluate on the function f(x, y) = sin(2πx)cos(2πy)
    f = lambda x: torch.sin(2 * torch.pi * x[0]) * torch.cos(2 * torch.pi * x[1])
    f_x = (
        lambda x: 2
        * torch.pi
        * torch.cos(2 * torch.pi * x[0])
        * torch.cos(2 * torch.pi * x[1])
    )
    f_y = (
        lambda x: -2
        * torch.pi
        * torch.sin(2 * torch.pi * x[0])
        * torch.sin(2 * torch.pi * x[1])
    )
    N = 20

    # Test three models: (1) Chebyshev + Chebyshev, (2) Chebyshev + Fourier, (3) Fourier + Fourier
    interpolant_1 = SpectralInterpolationND(
        Ns=[N + 1, N + 1], bases=["chebyshev", "chebyshev"], domains=[(0, 1), (0, 1)]
    )
    name_1 = "Chebyshev + Chebyshev"
    interpolant_2 = SpectralInterpolationND(
        Ns=[N + 1, N], bases=["chebyshev", "fourier"], domains=[(0, 1), (0, 1)]
    )
    name_2 = "Chebyshev + Fourier"
    interpolant_3 = SpectralInterpolationND(
        Ns=[N, N], bases=["fourier", "fourier"], domains=[(0, 1), (0, 1)]
    )
    name_3 = "Fourier + Fourier"

    # Create plots directory
    plot_dir = os.path.join(os.path.dirname(__file__), "../../plots/dev/interpolant_nd")
    os.makedirs(plot_dir, exist_ok=True)

    # Store errors for plotting
    errors = {
        "tensor_interpolation": [],
        "batch_interpolation": [],
        "node_interpolation": [],
        "spectral_derivative": [],
        "fd_derivative": [],
    }
    k_values = [1, 2, 3, 4, 5]  # FD stencil sizes to test

    for interpolant, name in zip(
        [interpolant_1, interpolant_2, interpolant_3], [name_1, name_2, name_3]
    ):
        print(f"\nTesting {name}")

        # Set the values at the nodes
        f_vals = f(
            torch.meshgrid(
                *[interpolant.nodes[i] for i in range(interpolant.n_dim)],
                indexing="ij",
            )
        )
        print(f"f_vals shape: {f_vals.shape}")
        print(f"interpolant.values shape: {interpolant.values.shape}")
        interpolant.values.data = f_vals.reshape(interpolant.values.shape)

        # 1. Test interpolation at grid of points
        N_dense = 100
        x_eval_dense = torch.meshgrid(
            *[torch.linspace(0, 1, N_dense) for _ in range(interpolant.n_dim)],
            indexing="ij",
        )
        f_eval_dense = interpolant(
            [torch.linspace(0, 1, N_dense) for _ in range(interpolant.n_dim)]
        )
        f_vals_dense = f(x_eval_dense)
        dense_error = torch.norm(f_eval_dense - f_vals_dense)
        errors["tensor_interpolation"].append(dense_error.item())
        print(f"Tensor interpolation error: {dense_error:.2e}")

        # 1b. Test interpolation at nodes
        f_eval_nodes = interpolant(interpolant.nodes)
        print(f"f_eval_nodes shape: {f_eval_nodes.shape}")
        print(f"f_eval_nodes error: {torch.norm(f_eval_nodes - f_vals)}")
        node_error = torch.norm(f_eval_nodes - f_vals)
        errors["node_interpolation"].append(node_error.item())
        print(f"Node interpolation error: {node_error:.2e}")

        # 2. Test batch interpolation
        x_eval_batch = torch.meshgrid(
            *[torch.linspace(0, 1, N_dense) for _ in range(interpolant.n_dim)],
            indexing="ij",
        )
        x_eval_batch = (
            torch.stack(x_eval_batch, dim=0)
            .permute(1, 0, 2)
            .reshape(N_dense**interpolant.n_dim, interpolant.n_dim)
        )
        f_eval_batch = torch.stack(
            [f(x_eval_batch[i]) for i in range(x_eval_batch.shape[0])]
        )
        interpolant_eval_batch = interpolant(x_eval_batch)
        batch_error = torch.norm(interpolant_eval_batch - f_eval_batch)
        errors["batch_interpolation"].append(batch_error.item())
        print(f"Batch interpolation error: {batch_error:.2e}")

        # 3. Test spectral derivatives
        interpolant_dx = interpolant.derivative(x_eval_batch, (1, 0))
        interpolant_dy = interpolant.derivative(x_eval_batch, (0, 1))
        f_dx = torch.stack([f_x(x_eval_batch[i]) for i in range(x_eval_batch.shape[0])])
        f_dy = torch.stack([f_y(x_eval_batch[i]) for i in range(x_eval_batch.shape[0])])
        spectral_error = torch.norm(interpolant_dx - f_dx) + torch.norm(
            interpolant_dy - f_dy
        )
        errors["spectral_derivative"].append(spectral_error.item())
        print(f"Spectral derivative error: {spectral_error:.2e}")

        # 4. Test FD derivatives with varying k
        fd_errors = []
        fd_residuals = []  # Store residuals for plotting
        for k in k_values:
            interpolant.update_fd_params(fd_k=[k, k])
            fd_dx = interpolant.derivative(x_eval_batch, (1, 0), use_spectral=False)
            fd_dy = interpolant.derivative(x_eval_batch, (0, 1), use_spectral=False)

            # Compute exact derivatives
            f_dx = torch.stack(
                [f_x(x_eval_batch[i]) for i in range(x_eval_batch.shape[0])]
            )
            f_dy = torch.stack(
                [f_y(x_eval_batch[i]) for i in range(x_eval_batch.shape[0])]
            )

            # Store residuals
            fd_residuals.append(
                {
                    "k": k,
                    "x": x_eval_batch[:, 0].detach().cpu().numpy(),
                    "y": x_eval_batch[:, 1].detach().cpu().numpy(),
                    "dx_residual": (fd_dx - f_dx).detach().cpu().numpy(),
                    "dy_residual": (fd_dy - f_dy).detach().cpu().numpy(),
                    "dx_exact": f_dx.detach().cpu().numpy(),
                    "dy_exact": f_dy.detach().cpu().numpy(),
                    "dx_computed": fd_dx.detach().cpu().numpy(),
                    "dy_computed": fd_dy.detach().cpu().numpy(),
                }
            )

            fd_error = torch.norm(fd_dx - f_dx) + torch.norm(fd_dy - f_dy)
            fd_errors.append(fd_error.item())
            print(f"k = {k}, FD error = {fd_error:.2e}")
        errors["fd_derivative"].append(fd_errors)
        print(f"FD derivative errors: {[f'{e:.2e}' for e in fd_errors]}")

        # Create residual plots
        plt.figure(figsize=(20, 10))

        # Plot 1: Residuals for k = 1
        plt.subplot(2, 3, 1)
        k_idx = 0  # First k value
        plt.scatter(
            fd_residuals[k_idx]["x"],
            fd_residuals[k_idx]["y"],
            c=fd_residuals[k_idx]["dx_residual"],
            cmap="RdBu",
        )
        plt.colorbar(label="dx residual")
        plt.title(f"dx residuals (k = {k_values[k_idx]})")

        plt.subplot(2, 3, 2)
        plt.scatter(
            fd_residuals[k_idx]["x"],
            fd_residuals[k_idx]["y"],
            c=fd_residuals[k_idx]["dy_residual"],
            cmap="RdBu",
        )
        plt.colorbar(label="dy residual")
        plt.title(f"dy residuals (k = {k_values[k_idx]})")

        # Plot 2: Residuals for k = 3
        plt.subplot(2, 3, 3)
        k_idx = 2  # Middle k value
        plt.scatter(
            fd_residuals[k_idx]["x"],
            fd_residuals[k_idx]["y"],
            c=fd_residuals[k_idx]["dx_residual"],
            cmap="RdBu",
        )
        plt.colorbar(label="dx residual")
        plt.title(f"dx residuals (k = {k_values[k_idx]})")

        plt.subplot(2, 3, 4)
        plt.scatter(
            fd_residuals[k_idx]["x"],
            fd_residuals[k_idx]["y"],
            c=fd_residuals[k_idx]["dy_residual"],
            cmap="RdBu",
        )
        plt.colorbar(label="dy residual")
        plt.title(f"dy residuals (k = {k_values[k_idx]})")

        # Plot 3: Exact vs computed derivatives
        plt.subplot(2, 3, 5)
        plt.scatter(
            fd_residuals[k_idx]["dx_exact"],
            fd_residuals[k_idx]["dx_computed"],
            alpha=0.5,
            label="dx",
        )
        plt.scatter(
            fd_residuals[k_idx]["dy_exact"],
            fd_residuals[k_idx]["dy_computed"],
            alpha=0.5,
            label="dy",
        )
        plt.plot([-10, 10], [-10, 10], "k--", label="y=x")
        plt.xlabel("Exact derivative")
        plt.ylabel("Computed derivative")
        plt.title("Exact vs Computed Derivatives")
        plt.legend()

        # Plot 4: Error distribution
        plt.subplot(2, 3, 6)
        plt.hist(fd_residuals[k_idx]["dx_residual"], bins=50, alpha=0.5, label="dx")
        plt.hist(fd_residuals[k_idx]["dy_residual"], bins=50, alpha=0.5, label="dy")
        plt.xlabel("Error")
        plt.ylabel("Count")
        plt.title("Error Distribution")
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(plot_dir, f"fd_residuals_{name.replace(' ', '_')}.png")
        )
        plt.close()

    # Create plots
    plt.figure(figsize=(15, 10))

    # Plot 1: Basic errors for all three models
    plt.subplot(2, 2, 1)
    x = np.arange(3)
    width = 0.2
    plt.bar(
        x - width, errors["tensor_interpolation"], width, label="Tensor Interpolation"
    )
    plt.bar(x, errors["node_interpolation"], width, label="Node Interpolation")
    plt.bar(
        x + width, errors["batch_interpolation"], width, label="Batch Interpolation"
    )
    plt.bar(
        x + width, errors["spectral_derivative"], width, label="Spectral Derivative"
    )
    plt.xticks(x, [name_1, name_2, name_3])
    plt.yscale("log")
    plt.ylabel("Error")
    plt.title("Basic Errors for Different Models")
    plt.legend()

    # Plot 2: FD convergence for each model
    plt.subplot(2, 2, 2)
    for i, (name, fd_errors) in enumerate(
        zip([name_1, name_2, name_3], errors["fd_derivative"])
    ):
        plt.semilogy(k_values, fd_errors, "o-", label=name)
    plt.xlabel("Stencil size k")
    plt.ylabel("Error")
    plt.title("FD Derivative Convergence")
    plt.legend()
    plt.grid(True, which="both", ls="-")

    # Plot 3: Error comparison between spectral and FD
    plt.subplot(2, 2, 3)
    spectral_errors = np.array(errors["spectral_derivative"])
    fd_errors = np.array([min(errs) for errs in errors["fd_derivative"]])
    plt.bar(x - width / 2, spectral_errors, width, label="Spectral")
    plt.bar(x + width / 2, fd_errors, width, label="Best FD")
    plt.xticks(x, [name_1, name_2, name_3])
    plt.yscale("log")
    plt.ylabel("Error")
    plt.title("Spectral vs Best FD Error")
    plt.legend()

    # Plot 4: FD convergence rate
    plt.subplot(2, 2, 4)
    for i, (name, fd_errors) in enumerate(
        zip([name_1, name_2, name_3], errors["fd_derivative"])
    ):
        rates = np.diff(np.log(fd_errors)) / np.diff(np.log(k_values))
        plt.semilogx(k_values[:-1], rates, "o-", label=name)
    plt.xlabel("Stencil size k")
    plt.ylabel("Convergence Rate")
    plt.title("FD Convergence Rate")
    plt.legend()
    plt.grid(True, which="both", ls="-")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "convergence_analysis.png"))
    plt.close()

    print(f"\nPlots saved to {plot_dir}/convergence_analysis.png")

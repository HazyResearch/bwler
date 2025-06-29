# -------------------------------------------------------------------------
#  fornberg.py  ––  Finite‑difference differentiation matrix on an
#                    arbitrary 1‑D grid using Fornberg's recursion
#
#  Given
#      alpha  : 1‑D NumPy array of N+1 distinct nodes  (α₀ … α_N)
#      k      : half‑bandwidth of the local stencil  (2k+1 points max)
#      m      : derivative order  (m = 0 → interpolation, 1 → f′, …)
#
#  the helper  D_banded_fornberg(alpha, k, m)
#  returns a dense (N+1)×(N+1) matrix  D  such that
#
#          f^{(m)}(α_i)   ≈   Σ_j  D[i, j] · f(α_j)
#
#  Every row is built by
#      • selecting the centred stencil   {α_{i-k}, …, α_{i+k}}
#        (clipped at the boundaries), and
#      • calling the three‑index recursion from
#
#          B. Fornberg,
#          "Generation of finite difference formulas on arbitrarily spaned grids,"
#          *J. Comput. Phys.* **38** (1980), 400‑407.
#
#  The routine reproduces the exact first, second, … derivative of the
#  polynomial of degree ≤ 2k on *any* irregular grid and works up to
#  a few hundred nodes in double precision.
#
#  Example
#  -------
#      >>> import numpy as np
#      >>> from fornberg import D_banded_fornberg
#      >>> x = torch.tensor([-1.0, 0.0, 1.0])          # three nodes
#      >>> D = D_banded_fornberg(x, k=1, m=1)      # 3‑point, first derivative
#      >>> print(D)
#      [[-1.     1.     0.   ]
#       [-0.5    0.     0.5  ]
#       [ 0.    -1.     1.   ]]
#
#  ------------------------------------------------------------------------

import numpy as np
import torch
import matplotlib.pyplot as plt
import os


# ──────────────────────────────────────────────────────────────────────────────
#  fornberg_full_table
#  -------------------
#  Construct the complete triangular weight table
#
#          δ⁽ᵐ⁾ₙ,ν     for  m = 0…M ,   n = 0…N ,   ν = 0…n
#
#  so that, after the routine finishes,
#
#          f^{(m)}(x0)  ≈  Σ_{ν=0}^{N}  δ⁽ᵐ⁾_{N,ν} · f(α_ν) .
#
#  Indices
#  --------
#      m   derivative order (row in the first axis of `delta`)
#      n   current stencil size minus one  — outer loop index
#      ν   node index inside the stencil  — inner loop index
#
#  Parameters
#  ----------
#      M      int        highest derivative order required  (M ≥ 0)
#      x0     float      point at which derivatives are desired
#      alpha  torch.Tensor  grid nodes α₀ … α_N  (distinct)
#
#  Returns
#  -------
#      delta  torch.Tensor, shape (M+1, N+1, N+1)
#             delta[m, n, ν] = δ⁽ᵐ⁾ₙ,ν
#             The useful weights for the full stencil are `delta[m, N, :]`.
#
# ──────────────────────────────────────────────────────────────────────────────
def fornberg_full_table(M: int, x0: float, alpha: torch.Tensor):
    assert alpha.ndim == 1
    N = len(alpha) - 1
    delta = torch.zeros((M + 1, N + 1, N + 1), device=alpha.device)  # (m, n, ν)
    delta[0, 0, 0] = 1.0
    c1 = 1.0
    for n in range(1, N + 1):
        c2 = 1.0
        for nu in range(n):
            c3 = alpha[n] - alpha[nu]
            c2 *= c3
            for m in range(0, min(n, M) + 1):
                term1 = (alpha[n] - x0) * delta[m, n - 1, nu]
                term2 = 0.0 if m == 0 else m * delta[m - 1, n - 1, nu]
                delta[m, n, nu] = (term1 - term2) / c3
        for m in range(0, min(n, M) + 1):
            termA = 0.0 if m == 0 else m * delta[m - 1, n - 1, n - 1]
            termB = (alpha[n - 1] - x0) * delta[m, n - 1, n - 1]
            delta[m, n, n] = (c1 / c2) * (termA - termB)
        c1 = c2
    return delta


# ──────────────────────────────────────────────────────────────────────────────
#  row_weights
#  ------------
#  Build one row of D^(m) using a 2k+1 stencil centred at node i
#
#  Parameters
#  ----------
#  alpha  torch.Tensor  grid nodes α₀ … α_N  (distinct)
#  i      int           node index
#  k      int           half‑bandwidth of the local stencil  (2k+1 points max)
#  m      int           derivative order  (m = 0 → interpolation, 1 → f′, …)
#
#  Returns
#  -------
#      w  torch.Tensor, shape (N+1,)
#
# ──────────────────────────────────────────────────────────────────────────────
def row_weights(alpha: torch.Tensor, i: int, k: int, m: int):
    assert alpha.ndim == 1
    N = len(alpha) - 1
    lo = max(0, i - k)
    hi = min(N, i + k)
    idx = np.arange(lo, hi + 1)
    table = fornberg_full_table(m, alpha[i], alpha[idx])
    w = torch.zeros(N + 1, device=alpha.device)
    w[idx] = table[m, -1, :]  # always take row n = len(idx)-1
    return w


# ──────────────────────────────────────────────────────────────────────────────
#  D_banded_fornberg
#  -----------------
#  Construct the banded finite‑difference differentiation matrix
#
#  Parameters
#  ----------
#      alpha  torch.Tensor  grid nodes α₀ … α_N  (distinct)
#      k      int           half‑bandwidth of the local stencil  (2k+1 points max)
#      m      int           derivative order  (m = 0 → interpolation, 1 → f′, …)
#
#  Returns
#  -------
#      D  torch.Tensor, shape (N+1, N+1)
#
# ──────────────────────────────────────────────────────────────────────────────
def D_banded_fornberg(alpha: torch.Tensor, k: int, m: int):
    assert alpha.ndim == 1
    N = len(alpha) - 1
    D = torch.zeros((N + 1, N + 1), device=alpha.device)
    for i in range(N + 1):
        D[i] = row_weights(alpha, i, k, m)
    return D


if __name__ == "__main__":
    import math

    torch.set_default_dtype(torch.float64)

    def cond(A, tol=1e-10):
        s = np.linalg.svd(A, compute_uv=False)
        # Filter out small singular values
        s = s[s > tol]
        return s[0] / s[-1]

    x = torch.tensor([-1.0, 0.0, 1.0])
    D = D_banded_fornberg(x, k=1, m=1)
    print(D)
    print(cond(D))

    # Grids
    def equi_nodes(N):
        return torch.linspace(-torch.pi, torch.pi, N + 1)

    def cheb_nodes(N):
        return torch.cos(torch.linspace(0, torch.pi, N + 1)) * torch.pi

    # Derivative orders to test
    Ms = [1, 2]

    # Test function and its derivatives
    def f(x):
        return torch.sin(x)

    def f_prime(x):
        return torch.cos(x)

    def f_prime2(x):
        return -torch.sin(x)

    # Sweep N
    Ns = [2**i + 1 for i in range(4, 8)]

    # Create plots directory
    plot_dir = os.path.join(os.path.dirname(__file__), "../../../plots/dev/fornberg")
    os.makedirs(plot_dir, exist_ok=True)

    for M in Ms:
        for N in Ns:
            # Sweep stencil sizes
            ks = [2**i for i in range(0, math.floor(math.log2(N)))]

            # Initialize arrays to store results
            equi_errors = []
            cheb_errors = []
            equi_conds = []
            cheb_conds = []

            # Test both node types
            for k in ks:
                # Equispaced nodes
                x_equi = equi_nodes(N)
                D_equi = D_banded_fornberg(x_equi, k=k, m=M)
                f_equi = f(x_equi)
                if M == 1:
                    f_exact = f_prime(x_equi)
                else:
                    f_exact = f_prime2(x_equi)
                f_approx = D_equi @ f_equi
                equi_errors.append(torch.norm(f_approx - f_exact).item())
                equi_conds.append(cond(D_equi.numpy()))

                # Chebyshev nodes
                x_cheb = cheb_nodes(N)
                D_cheb = D_banded_fornberg(x_cheb, k=k, m=M)
                f_cheb = f(x_cheb)
                if M == 1:
                    f_exact = f_prime(x_cheb)
                else:
                    f_exact = f_prime2(x_cheb)
                f_approx = D_cheb @ f_cheb
                cheb_errors.append(torch.norm(f_approx - f_exact).item())
                cheb_conds.append(cond(D_cheb.numpy()))

            # Create plots
            plt.figure(figsize=(15, 5))

            # Plot 1: Error vs stencil size
            plt.subplot(1, 2, 1)
            plt.loglog(ks, equi_errors, "o-", label="Equispaced")
            plt.loglog(ks, cheb_errors, "s-", label="Chebyshev")
            plt.xlabel("Stencil size (2k+1)")
            plt.ylabel("Error")
            plt.title(f"Approximation Error (N={N}, M={M})")
            plt.legend()
            plt.grid(True)

            # Plot 2: Condition number vs stencil size
            plt.subplot(1, 2, 2)
            plt.loglog(ks, equi_conds, "o-", label="Equispaced")
            plt.loglog(ks, cheb_conds, "s-", label="Chebyshev")
            plt.xlabel("Stencil size (2k+1)")
            plt.ylabel("Condition number")
            plt.title(f"Matrix Conditioning (N={N}, M={M})")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"fornberg_N{N}_M{M}.png"))
            plt.close()

            # Print summary
            print(f"\nResults for N={N}, M={M}:")
            print(
                "Stencil size | Equispaced Error | Chebyshev Error | Equispaced Cond | Chebyshev Cond"
            )
            print("-" * 80)
            for i, k in enumerate(ks):
                print(
                    f"{2*k+1:11d} | {equi_errors[i]:14.2e} | {cheb_errors[i]:14.2e} | {equi_conds[i]:14.2e} | {cheb_conds[i]:14.2e}"
                )

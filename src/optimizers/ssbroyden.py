import math
from typing import Iterable, List, Tuple, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required

# -----------------------------------------------------------------------------
# Helper utilities for flattening / un‑flattening parameter tensors
# -----------------------------------------------------------------------------

def _flatten(params: List[Tensor]) -> Tensor:
    """Flatten a list of tensors into a single 1‑D tensor (views, not copies)."""
    return torch.cat([p.data.view(-1) for p in params])


def _unflatten(vec: Tensor, params_example: List[Tensor]) -> List[Tensor]:
    """Reshape a flat vector back to the shapes of *params_example* (views)."""
    views: List[Tensor] = []
    offset = 0
    for p in params_example:
        numel = p.numel()
        views.append(vec[offset : offset + numel].view_as(p))
        offset += numel
    return views


# -----------------------------------------------------------------------------
# SSBroyden‑II optimizer with *strong Wolfe* line‑search
# -----------------------------------------------------------------------------

class SSBroyden2(Optimizer):
    """Dense *Self‑Scaled Broyden‑II* optimizer with a strong‑Wolfe line search.

    Notes
    -----
    * Stores a full inverse‑Hessian (\mathcal{O}(n²) mem) – only practical for
      small/medium models.
    * Each `step()` may call the `closure` many times due to the line search, so
      **the closure *must* recompute the loss *and* its gradient** (i.e. run a
      forward + backward pass).
    * Based on the algorithm in Urbán *et al.*, 2025 and the reference
      ``modified_optimize.py`` code provided by the user.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1.0,
        *,
        init_scale: bool = False,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_ls: int = 20,
        device: Optional[torch.device] = None,
    ):
        if lr <= 0:
            raise ValueError("`lr` (initial step‑length guess) must be > 0.")
        if not (0 < c1 < c2 < 1):
            raise ValueError("Require 0 < c1 < c2 < 1 for Wolfe conditions.")

        defaults = dict(
            lr=lr,
            init_scale=init_scale,
            c1=c1,
            c2=c2,
            max_ls=max_ls,
        )
        super().__init__(params, defaults)

        # Parameter bookkeeping (single group expected in most research code)
        group = self.param_groups[0]
        self._params = group["params"]
        self._device = device or self._params[0].device

        # Inverse Hessian estimate – created lazily once we know parameter count
        self.state["H"] = None  # type: ignore[index]

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _pack_params(self) -> Tensor:
        return _flatten(self._params).detach()

    def _pack_grads(self) -> Tensor:
        grads: List[Tensor] = []
        for p in self._params:
            if p.grad is None:
                grads.append(torch.zeros_like(p.data.view(-1)))
            else:
                grads.append(p.grad.view(-1))
        return torch.cat(grads).detach()

    def _write_params(self, flat: Tensor):
        for view, p in zip(_unflatten(flat, self._params), self._params):
            p.data.copy_(view)

    def _zero_grad(self):
        for p in self._params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    # ------------------------------------------------------------------
    # Strong Wolfe line‑search (backtracking variant, no *zoom*)
    # ------------------------------------------------------------------
    def _wolfe_line_search(
        self,
        closure,
        xk: Tensor,
        fk: Tensor,
        gk: Tensor,
        pk: Tensor,
        gk_dot_pk: Tensor,
        c1: float,
        c2: float,
        alpha0: float,
        max_ls: int,
    ) -> Tuple[float, Tensor, Tensor]:
        """Return (alpha, f_new, g_new) that satisfy strong Wolfe conditions."""

        alpha = alpha0
        prev_f: Optional[Tensor] = None

        for _ in range(max_ls):
            # Evaluate at new trial point
            x_new = xk + alpha * pk
            self._write_params(x_new)
            self._zero_grad()
            with torch.enable_grad():
                f_new_tensor = closure()
            f_new = f_new_tensor.detach()
            g_new = self._pack_grads()
            g_new_dot_pk = g_new.dot(pk)

            # Armijo condition
            if f_new > fk + c1 * alpha * gk_dot_pk or (
                prev_f is not None and f_new >= prev_f
            ):
                alpha *= 0.5  # back‑track
                prev_f = f_new
                continue

            # Curvature condition
            if abs(g_new_dot_pk) <= c2 * abs(gk_dot_pk):
                return alpha, f_new, g_new

            # If slope is positive, step is too large – backtrack
            if g_new_dot_pk >= 0:
                alpha *= 0.5
            else:
                # Otherwise try expanding a bit (helps when pk poorly scaled)
                alpha *= 1.1
            prev_f = f_new

        # Fallback – return last tried values (may violate Wolfe)
        return alpha, f_new, g_new

    # ------------------------------------------------------------------
    # Main optimiser step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure):  # closure required!
        if closure is None:
            raise RuntimeError("SSBroyden2 requires a `closure` that re‑evaluates the model.")

        group = self.param_groups[0]
        c1: float = group["c1"]
        c2: float = group["c2"]
        max_ls: int = group["max_ls"]
        alpha0: float = group["lr"]  # initial guess for step length
        init_scale: bool = group["init_scale"]

        # ------------------------------------------------------------------
        # Current loss & gradient at x_k
        # ------------------------------------------------------------------
        self._zero_grad()
        with torch.enable_grad():
            loss_tensor = closure()
        f_k = loss_tensor.detach()
        x_k = self._pack_params()
        g_k = self._pack_grads()

        n = x_k.numel()

        # ------------------------------------------------------------------
        # Initialise inverse Hessian as (1/τ₀)·I on the very first iteration
        # ------------------------------------------------------------------
        if self.state["H"] is None:
            tau0 = 1.0
            if init_scale:
                grad_norm = g_k.norm()
                if grad_norm > 0:
                    tau0 = grad_norm.item()
            self.state["H"] = torch.eye(n, device=self._device) / tau0

        H_k: Tensor = self.state["H"]

        # ------------------------------------------------------------------
        # Compute search direction (ensure descent)
        # ------------------------------------------------------------------
        pk = -H_k @ g_k
        if pk.dot(g_k) >= 0:
            pk = -g_k  # fall back to steepest descent

        gk_dot_pk = g_k.dot(pk)

        # ------------------------------------------------------------------
        # Line search satisfying strong Wolfe conditions
        # ------------------------------------------------------------------
        alpha, f_new, g_new = self._wolfe_line_search(
            closure,
            x_k,
            f_k,
            g_k,
            pk,
            gk_dot_pk,
            c1,
            c2,
            alpha0,
            max_ls,
        )

        x_new = x_k + alpha * pk
        self._write_params(x_new)  # ensure params on final point

        # ------------------------------------------------------------------
        # Quasi‑Newton update (self‑scaled Broyden‑II)
        # ------------------------------------------------------------------
        s_k = x_new - x_k        # = alpha · p_k
        y_k = g_new - g_k

        rhok_inv = y_k.dot(s_k)
        rhok_inv = rhok_inv# + torch.sign(rhok_inv) * 1e-8  # ensure non‑zero
        if rhok_inv.abs() < 1e-30:
            # Skip rank‑two update if curvature condition degenerates
            self.state["H"] = H_k
            return f_new  # type: ignore[return‑value]
        rhok = 1.0 / rhok_inv

        Hkyk = H_k @ y_k
        ykHkyk = y_k.dot(Hkyk)
        h_k = ykHkyk * rhok

        b_k = -alpha * rhok * s_k.dot(g_k)
        a_k = b_k * h_k - 1.0

        # ρ_k^‑, θ_k, τ_k   (cf. Urbán et al.)
        rho_k_minus = torch.minimum(torch.tensor(1.0, device=self._device),
                                    h_k * (1 - torch.sqrt(torch.abs(a_k) / (1 + a_k))))
        theta_k_minus = (rho_k_minus - 1) / a_k
        theta_k_plus = 1 / rho_k_minus
        theta_k = torch.max(theta_k_minus, torch.min(theta_k_plus, (1 - b_k) / b_k))

        rho_k_cap = torch.minimum(torch.tensor(1.0, device=self._device), 1 / b_k)
        sigma_k = 1 + theta_k * a_k
        sigma_pow = torch.abs(sigma_k) ** (1.0 / (1 - n))
        if theta_k <= 0:
            tau_k = torch.minimum(rho_k_cap * sigma_pow, sigma_k)
        else:
            tau_k = rho_k_cap * torch.minimum(sigma_pow, 1 / theta_k)

        v_k = rhok * s_k - Hkyk / ykHkyk
        phi_k = (1 - theta_k) / (1 + a_k * theta_k)

        H_new = (
            (H_k - torch.outer(Hkyk, Hkyk) / ykHkyk + phi_k * ykHkyk * torch.outer(v_k, v_k))
            / tau_k
            + rhok * torch.outer(s_k, s_k)
        )

        self.state["H"] = H_new.detach().clone()

        return f_new
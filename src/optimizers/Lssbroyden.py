import torch
from torch.optim.optimizer import Optimizer
import math
from collections import deque
from typing import Optional, Callable, Tuple

class L_SSBroyden(Optimizer):
    """Limited-memory Self-Scaled Broyden optimizer.
    
    This implements a limited-memory version of the Self-Scaled Broyden-II algorithm,
    similar to how L-BFGS is a limited-memory version of BFGS.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1.0)
        history_size: number of vector pairs to store (default: 10)
        init_scale: whether to scale initial Hessian estimate (default: False)
        c1: Armijo condition parameter (default: 1e-4)
        c2: Wolfe condition parameter (default: 0.9)
        max_ls: maximum line search iterations (default: 20)
        tolerance_grad: gradient tolerance for convergence (default: 1e-7)
        tolerance_change: parameter change tolerance (default: 1e-9)
    """
    
    def __init__(self, params, lr=1.0, history_size=10, init_scale=False,
                 c1=1e-4, c2=0.9, max_ls=20, tolerance_grad=1e-30,
                 tolerance_change=1e-9):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if history_size <= 0:
            raise ValueError(f"Invalid history size: {history_size}")
        if not 0 < c1 < c2 < 1:
            raise ValueError("Require 0 < c1 < c2 < 1 for Wolfe conditions")
            
        defaults = dict(lr=lr, history_size=history_size, init_scale=init_scale,
                       c1=c1, c2=c2, max_ls=max_ls, tolerance_grad=tolerance_grad,
                       tolerance_change=tolerance_change)
        super().__init__(params, defaults)
        
        if len(self.param_groups) != 1:
            raise ValueError("L_SSBroyden doesn't support per-parameter options")
            
        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        
    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(p.numel() for p in self._params if p.requires_grad)
        return self._numel_cache
    
    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)
    
    def _gather_flat_params(self):
        views = []
        for p in self._params:
            view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)
    
    def _set_flat_params(self, flat_params):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view_as(p))
            offset += numel
            
    def _compute_Hy_recursive(self, y, history_s, history_y, history_tau, 
                             history_phi, history_rho, idx, gamma):
        """Recursively compute H_i * y without forming H_i explicitly."""
        if idx < 0:
            # Base case: H_0 = gamma * I
            return gamma * y
            
        # Recursive call for H_{i-1} * y
        z = self._compute_Hy_recursive(y, history_s, history_y, history_tau,
                                      history_phi, history_rho, idx-1, gamma)
        
        # Apply scaling
        if idx > 0:
            z = z / history_tau[idx-1]
        
        # Get history at index idx
        s_i = history_s[idx]
        y_i = history_y[idx]
        rho_i = history_rho[idx]
        phi_i = history_phi[idx]
        
        # Standard BFGS correction
        # z = z - (y_i · z)/(y_i · H_{i-1} y_i) * H_{i-1} y_i + rho_i * (s_i · z) * s_i
        
        # First compute H_{i-1} y_i
        H_y_i = self._compute_Hy_recursive(y_i, history_s, history_y, history_tau,
                                          history_phi, history_rho, idx-1, gamma)
        y_H_y = y_i.dot(H_y_i)
        
        if abs(y_H_y) > 1e-16:
            z = z - (y_i.dot(z) / y_H_y) * H_y_i
        
        z = z + rho_i * s_i.dot(z) * s_i
        
        # Self-scaled Broyden correction if phi != 1
        if abs(phi_i - 1.0) > 1e-16:
            # v_i = sqrt(y_H_y) * [s_i/(y_i·s_i) - H_y_i/y_H_y]
            v_i = math.sqrt(abs(y_H_y)) * (rho_i * s_i - H_y_i / y_H_y)
            v_norm_sq = v_i.dot(v_i)
            if v_norm_sq > 1e-16:
                z = z + phi_i * (v_i.dot(z) / v_norm_sq) * v_i
        
        return z
    
    def _two_loop_recursion(self, g, history_s, history_y, history_tau,
                           history_phi, history_rho, gamma):
        """L-SSBroyden two-loop recursion to compute H_k * g."""
        m = len(history_s)
        if m == 0:
            return gamma * g
        
        # Forward loop
        q = g.clone()
        alphas = []
        xis = []  # for Broyden corrections
        tau_cumulative = 1.0
        
        for i in range(m-1, -1, -1):
            tau_cumulative *= history_tau[i]
            rho_i = history_rho[i]
            s_i = history_s[i]
            y_i = history_y[i]
            phi_i = history_phi[i]
            
            # Standard BFGS-like correction
            alpha_i = rho_i * s_i.dot(q)
            q = q - alpha_i * y_i
            alphas.append(alpha_i)
            
            # Self-scaled Broyden correction
            if abs(phi_i - 1.0) > 1e-16:
                # Compute H_i y_i
                H_y_i = self._compute_Hy_recursive(y_i, history_s[:i], history_y[:i],
                                                  history_tau[:i], history_phi[:i],
                                                  history_rho[:i], i-1, gamma)
                h_i = y_i.dot(H_y_i)
                
                if abs(h_i) > 1e-16:
                    v_i = math.sqrt(abs(h_i)) * (rho_i * s_i - H_y_i / h_i)
                    v_norm_sq = v_i.dot(v_i)
                    if v_norm_sq > 1e-16:
                        xi_i = v_i.dot(q) / v_norm_sq
                        q = q - xi_i * v_i
                        xis.append((i, xi_i, v_i))
        
        # Apply scaling
        r = (gamma / tau_cumulative) * q
        
        # Backward loop
        alphas.reverse()
        for i in range(m):
            rho_i = history_rho[i]
            s_i = history_s[i]
            y_i = history_y[i]
            
            # Standard BFGS-like correction
            beta_i = rho_i * y_i.dot(r)
            r = r + (alphas[m-1-i] - beta_i) * s_i
        
        # Apply Broyden corrections in backward loop
        for i, xi_i, v_i in reversed(xis):
            eta_i = v_i.dot(r) / v_i.dot(v_i)
            r = r + (xi_i - eta_i) * v_i
        
        return r
    
    def _compute_tau_phi(self, s_k, y_k, grad_k, alpha_k, history_s, history_y, 
                        history_tau, history_phi, history_rho, gamma):
        """Compute self-scaling parameters tau_k and phi_k."""
        # Compute auxiliary variables
        rho_k_inv = y_k.dot(s_k)
        if abs(rho_k_inv) < 1e-10:
            return 1.0, 1.0  # Skip update if y·s is too small
        
        rho_k = 1.0 / rho_k_inv
        
        # b_k = s_k · H_k^{-1} s_k / (y_k · s_k) = -alpha_k * s_k · grad_k / (y_k · s_k)
        b_k = -alpha_k * s_k.dot(grad_k) / rho_k_inv
        
        # Compute H_k * y_k using the two-loop recursion
        if len(history_s) > 0:
            H_k_y_k = self._two_loop_recursion(
                y_k,
                list(history_s),
                list(history_y),
                list(history_tau),
                list(history_phi),
                list(history_rho),
                gamma
            )
        else:
            H_k_y_k = gamma * y_k
        
        # h_k = y_k · H_k y_k / (y_k · s_k)
        h_k = y_k.dot(H_k_y_k) * rho_k
        
        # Compute tau and phi according to the paper
        a_k = h_k * b_k - 1
        
        if abs(a_k) < 1e-16:
            return 1.0, 1.0
        
        # Self-scaled parameters
        rho_k_minus = min(1.0, h_k * (1 - math.sqrt(abs(a_k) / (1 + a_k))))
        theta_k_minus = (rho_k_minus - 1) / a_k
        theta_k_plus = 1 / rho_k_minus
        theta_k = max(theta_k_minus, min(theta_k_plus, (1 - b_k) / b_k))
        
        rho_k_cap = min(1.0, 1 / b_k) if b_k > 0 else 1.0
        sigma_k = 1 + theta_k * a_k
        n = self._numel()
        if n > 20:  # For large n, use approximation
            log_sigma = torch.log(abs(sigma_k) + 1e-10)
            sigma_pow = torch.exp(log_sigma / (1 - n))
        else:
            sigma_pow = abs(sigma_k) ** (1.0 / (1 - n))
            
        if theta_k <= 0:
            tau_k = min(rho_k_cap * sigma_pow, sigma_k)
        else:
            tau_k = rho_k_cap * min(sigma_pow, 1 / theta_k)
            
            phi_k = (1 - theta_k) / (1 + a_k * theta_k)
        
        # Ensure tau_k is positive and reasonable
        tau_k = max(1e-8, min(tau_k, 1e8))
        
        return tau_k, phi_k
    
    def _line_search_strong_wolfe(self, closure, x_k, f_k, g_k, p_k, g_dot_p, c1, c2, max_ls):
        """Strong Wolfe line search."""
        alpha = 1.0  # Initial step size
        prev_f = None
        
        for _ in range(max_ls):
            # Evaluate at new point
            x_new = x_k + alpha * p_k
            self._set_flat_params(x_new)
            
            with torch.enable_grad():
                f_new = closure()
            g_new = self._gather_flat_grad()
            g_new_dot_p = g_new.dot(p_k)
            
            # Armijo condition
            if f_new > f_k + c1 * alpha * g_dot_p or (prev_f is not None and f_new >= prev_f):
                alpha *= 0.5
                prev_f = f_new
                continue
            
            # Curvature condition
            if abs(g_new_dot_p) <= c2 * abs(g_dot_p):
                return alpha, f_new, g_new
            
            # If slope is positive, step is too large
            if g_new_dot_p >= 0:
                alpha *= 0.5
            else:
                alpha *= 1.1
            prev_f = f_new
        
        # Return last tried values
        return alpha, f_new, g_new
    
    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            raise RuntimeError("L_SSBroyden requires a closure")
        
        group = self.param_groups[0]
        lr = group['lr']
        c1 = group['c1']
        c2 = group['c2']
        max_ls = group['max_ls']
        history_size = group['history_size']
        init_scale = group['init_scale']
        
        # Get state
        state = self.state[id(self)]
        if len(state) == 0:
            # Initialize state
            state['n_iter'] = 0
            state['history_s'] = deque(maxlen=history_size)
            state['history_y'] = deque(maxlen=history_size)
            state['history_tau'] = deque(maxlen=history_size)
            state['history_phi'] = deque(maxlen=history_size)
            state['history_rho'] = deque(maxlen=history_size)
            state['gamma'] = 1.0
        
        # Increment iteration counter
        state['n_iter'] += 1
        
        # Evaluate initial loss and gradient
        with torch.enable_grad():
            f_k = closure()
        x_k = self._gather_flat_params()
        g_k = self._gather_flat_grad()

        if g_k.norm() < group['tolerance_grad']:
            return f_k

        
        # Compute search direction using two-loop recursion
        if len(state['history_s']) == 0:
            # First iteration: use scaled gradient descent
            if init_scale:
                grad_norm = g_k.norm()
                if grad_norm > 0:
                    state['gamma'] = 1.0 / grad_norm.item()
            p_k = -state['gamma'] * g_k
        else:
            # Use L-SSBroyden two-loop recursion
            p_k = -self._two_loop_recursion(
                g_k,
                list(state['history_s']),
                list(state['history_y']),
                list(state['history_tau']),
                list(state['history_phi']),
                list(state['history_rho']),
                state['gamma']
            )
        
        # Check if direction is descent
        g_dot_p = g_k.dot(p_k)
        if g_dot_p >= 0:
            p_k = -g_k
            g_dot_p = -g_k.dot(g_k)
        
        # Line search
        alpha, f_new, g_new = self._line_search_strong_wolfe(
            closure, x_k, f_k, g_k, p_k, g_dot_p, c1, c2, max_ls
        )
        
        # Update parameters
        x_new = x_k + alpha * p_k
        self._set_flat_params(x_new)
        
        # Compute updates for history
        s_k = x_new - x_k  # = alpha * p_k
        y_k = g_new - g_k
        
        rho_k_inv = y_k.dot(s_k)
        if abs(rho_k_inv) > 1e-16:
            rho_k = 1.0 / rho_k_inv
            
            # Compute H_k y_k for tau and phi calculation
            if len(state['history_s']) > 0:
                H_k_y_k = self._two_loop_recursion(
                    y_k,
                    list(state['history_s']),
                    list(state['history_y']),
                    list(state['history_tau']),
                    list(state['history_phi']),
                    list(state['history_rho']),
                    state['gamma']
                )
            else:
                H_k_y_k = state['gamma'] * y_k
            
            # Compute self-scaling parameters
            tau_k, phi_k = self._compute_tau_phi(
                s_k, y_k, g_k, alpha,
                list(state['history_s']),
                list(state['history_y']),
                list(state['history_tau']),
                list(state['history_phi']),
                list(state['history_rho']),
                state['gamma']
            )
            
            # Update history
            state['history_s'].append(s_k)
            state['history_y'].append(y_k)
            state['history_tau'].append(tau_k)
            state['history_phi'].append(phi_k)
            state['history_rho'].append(rho_k)
            
            # Update gamma (similar to L-BFGS)
            state['gamma'] = rho_k_inv / y_k.dot(y_k)
        
        return f_new
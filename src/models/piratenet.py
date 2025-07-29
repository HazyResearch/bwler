import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict, Tuple, Union
# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class PeriodEmbs(nn.Module):
    """Periodic embeddings module (cos/sin pairs)."""
    def __init__(self, period: Tuple[float, ...], axis: Tuple[int, ...],
                 trainable: Tuple[bool, ...]):
        super().__init__()
        self.axis = axis
        for idx, (p, tr) in enumerate(zip(period, trainable)):
            name = f"period_{idx}"
            tensor = torch.tensor(float(p))
            if tr:
                self.register_parameter(name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
        if isinstance(x, torch.Tensor):
            xs = torch.unbind(x, dim=-1)
        else:
            xs = x
        out = []
        for dim, xi in enumerate(xs):
            if dim in self.axis:
                idx = self.axis.index(dim)
                p = getattr(self, f"period_{idx}")
                out.extend([torch.cos(p * xi), torch.sin(p * xi)])
            else:
                out.append(xi)
        return torch.cat(out, dim=-1)                        # << CAT not stack

class FourierEmbs(nn.Module):
    """Gaussian-random Fourier features."""
    def __init__(self, embed_scale: float, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_scale = embed_scale
        self.register_buffer("kernel", None)

    def forward(self, x: torch.Tensor):
        if self.kernel is None:                             # first call → create
            k = torch.randn(x.size(-1), self.embed_dim // 2,
                             device=x.device, dtype=x.dtype) * self.embed_scale
            self.register_buffer("kernel", k)               # keep as buffer
        proj = x @ self.kernel
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

class Embedding(nn.Module):
    def __init__(self, periodicity: Optional[Dict] = None,
                 fourier_emb: Optional[Dict] = None):
        super().__init__()
        self.period_emb = PeriodEmbs(**periodicity) if periodicity else None
        self.fourier_emb = FourierEmbs(**fourier_emb) if fourier_emb else None

    def forward(self, x):
        if self.period_emb:  x = self.period_emb(x)
        if self.fourier_emb: x = self.fourier_emb(x)
        return x
# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
class Dense(nn.Module):
    """Fully-connected layer with optional weight-factorisation re-param."""
    def __init__(self, in_features, out_features, reparam=None):
        super().__init__()
        if reparam and reparam.get("type") == "weight_fact":
            self.V = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.xavier_normal_(self.V)
            mu, sigma = reparam.get("mean", 1.), reparam.get("stddev", .1)
            self.log_s = nn.Parameter(torch.randn(out_features)*sigma + mu)
            self.bias  = nn.Parameter(torch.zeros(out_features))
            self.reparam = True
        else:
            self.linear = nn.Linear(in_features, out_features)
            nn.init.xavier_normal_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
            self.reparam = False

    def forward(self, x):
        if self.reparam:
            W = torch.exp(self.log_s).unsqueeze(1) * self.V
            return F.linear(x, W, self.bias)
        return self.linear(x)
# --------------------------------------------------------------------------- #
# Bottleneck block
# --------------------------------------------------------------------------- #
_ACTIVATIONS = dict(relu=F.relu, gelu=F.gelu, silu=F.silu, sigmoid=torch.sigmoid,
                    tanh=torch.tanh, sin=torch.sin)

class PIModifiedBottleneck(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int,
                 activation: str = "tanh", nonlinearity: float = 0.,
                 reparam: Optional[Dict] = None):
        super().__init__()
        self.hidden_dim, self.output_dim = hidden_dim, output_dim
        self.reparam = reparam
        self.act = _ACTIVATIONS.get(activation, torch.tanh)

        self.dense1 = self.dense2 = self.dense3 = None
        # residual mixing coefficient – initialised from `nonlinearity`
        self.alpha = nn.Parameter(torch.tensor(float(nonlinearity)))

    def _init_layers(self, in_dim):
        if self.dense1 is None:
            self.dense1 = Dense(in_dim,  self.hidden_dim, self.reparam)
            self.dense2 = Dense(self.hidden_dim, self.hidden_dim, self.reparam)
            self.dense3 = Dense(self.hidden_dim, self.output_dim, self.reparam)

    def _mix(self, coeff, u, v):
        """coeff in (0,1) after sigmoid."""
        coeff = torch.sigmoid(coeff)
        return coeff * u + (1.0 - coeff) * v

    def forward(self, x, u, v):
        self._init_layers(x.size(-1))
        skip = x                                        # save identity

        x = self.act(self.dense1(x))
        x = self._mix(x, u, v)

        x = self.act(self.dense2(x))
        x = self._mix(x, u, v)

        x = self.act(self.dense3(x))

        # residual connection with sigmoid-bounded alpha
        res_w = torch.sigmoid(self.alpha)
        x = res_w * x + (1.0 - res_w) * skip
        return x
# --------------------------------------------------------------------------- #
# PirateNet
# --------------------------------------------------------------------------- #
class PirateNet(nn.Module):
    def __init__(self, num_layers=2, hidden_dim=256, out_dim=1,
                 activation="tanh", nonlinearity=0.,
                 periodicity=None, fourier_emb=None, reparam=None,
                 pi_init: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_layers, self.hidden_dim = num_layers, hidden_dim
        self.out_dim, self.reparam = out_dim, reparam
        self.act = _ACTIVATIONS.get(activation, torch.tanh)
        self.embed = Embedding(periodicity, fourier_emb)

        self.u_dense = self.v_dense = None
        self.blocks   = nn.ModuleList()
        self.out_layer = None
        self.register_parameter("pi_init_kernel",
                                nn.Parameter(pi_init.clone(), requires_grad=False)
                                if pi_init is not None else None)

    # -------------------------
    def _init_layers(self, in_dim):
        if self.u_dense is not None:           # already built
            return
        self.u_dense = Dense(in_dim, self.hidden_dim, self.reparam)
        self.v_dense = Dense(in_dim, self.hidden_dim, self.reparam)

        cur_dim = in_dim
        for _ in range(self.num_layers):
            self.blocks.append(PIModifiedBottleneck(
                hidden_dim=self.hidden_dim, output_dim=cur_dim,
                activation='tanh', nonlinearity=0., reparam=self.reparam))
        if self.pi_init_kernel is None:
            self.out_layer = Dense(cur_dim, self.out_dim, self.reparam)

    # -------------------------
    def forward(self, x):
        x = self.embed(x)
        self._init_layers(x.size(-1))

        u = self.act(self.u_dense(x))
        v = self.act(self.v_dense(x))

        for blk in self.blocks:
            x = blk(x, u, v)

        x_feat = self.act(x)                               # final non-linearity

        if self.pi_init_kernel is not None:
            y = torch.matmul(x_feat, self.pi_init_kernel)
        else:
            y = self.out_layer(x_feat)
        return x_feat, y
# --------------------------------------------------------------------------- #

import torch


def l2_error(
    u_eval: torch.Tensor, u_true: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(u_eval)
    return (torch.sum((u_eval - u_true) ** 2 * mask) / torch.sum(mask)).item()


def max_error(
    u_eval: torch.Tensor, u_true: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(u_eval)
    return torch.max(torch.abs(u_eval - u_true) * mask).item()


def l2_relative_error(
    u_eval: torch.Tensor, u_true: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(u_eval)
    return (
        torch.norm((u_eval - u_true) * mask, p=2) / torch.norm(u_true * mask, p=2)
    ).item()

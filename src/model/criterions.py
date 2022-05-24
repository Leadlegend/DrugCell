import torch
import torch.nn as nn
import torch.nn.functional as F


def Pearson_Correlation(y_pred: torch.Tensor,
                        y_true: torch.Tensor) -> torch.Tensor:
    """
    y_pred: [N, ]
    y_true: [N, ]
    """
    yp = y_pred - torch.mean(y_pred)
    yt = y_true - torch.mean(y_true)

    return torch.sum(yp * yt) / (torch.norm(yp, 2) * torch.norm(yt, 2))


def DrugCellLoss(outputs: dict, y_true: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    criterion = nn.MSELoss()
    for name, y_pred in outputs.items():
        if name == 'final':
            loss += criterion(y_pred, y_true)
        else:
            loss += 0.2 * criterion(y_pred, y_true)
    return loss


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort().to(x.device)
    ranks = torch.zeros_like(tmp, device=x.device)
    ranks[tmp] = torch.arange(len(x)).to(x.device)
    return ranks


def Spearman_Correlation(y_pred: torch.Tensor,
                         y_true: torch.Tensor) -> torch.Tensor:
    """
    Non-differentiable version of Spearman Correlation
    y_pred: [N, ]
    y_true: [N, ]
    """
    if y_pred.shape[-1] == 1:
        y_pred.squeeze_(-1)
    if y_true.shape[-1] == 1:
        y_true.squeeze_(-1)

    yp_rank = _get_ranks(y_pred)
    yt_rank = _get_ranks(y_true)

    n = yp_rank.size(0)
    upper = 6 * torch.sum((yp_rank - yt_rank).pow(2))
    down = n * (n**2 - 1.0)
    return 1.0 - (upper / down)

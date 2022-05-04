import torch
import torch.nn as nn
import torch.nn.functional as F

def Pearson_Correlation(x, y):
    xx = x - torch.mean(x)
    yy = y - torch.mean(y)

    return torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy, 2))


def DrugCellLoss(outputs, y_true):
    loss = 0.0
    criterion = nn.MSELoss()
    for name, y_pred in outputs.items():
        if name == 'final':
            loss += criterion(y_pred, y_true)
        else:
            loss += 0.2 * criterion(y_pred, y_true)
    return loss


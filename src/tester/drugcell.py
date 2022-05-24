import torch

from .base import Tester


class DrugCellTester(Tester):
    def __init__(self, model, criterion, config, device,
                 data_loader, epoch_criterion=None):
        super(DrugCellTester, self).__init__(model, criterion, config, device,
                 data_loader, epoch_criterion)
        self.model._to_(self.device)

    def _concat_output(self, labels, output):
        if not labels.size()[0]:
            labels = output.detach()
        else:
            labels = torch.cat([labels, output.detach()], dim=0)
        return labels

    def _out2pred(self, output):
        pred, _ = output
        return pred
    
    def _update_label(self, y_pred, y_gold, labels_pred, labels_gold):
        return self._concat_output(labels_pred, y_pred['final']), self._concat_output(labels_gold, y_gold)


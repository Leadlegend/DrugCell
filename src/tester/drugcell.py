import os
import torch

from tqdm import tqdm
from .base import Tester


class DrugCellTester(Tester):
    def __init__(self, model, config, device,
                 data_loader, criterion=None, epoch_criterion=None):
        super(DrugCellTester, self).__init__(model, config, device,
                                             data_loader, criterion, epoch_criterion)
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

    def rlipp_preprocess(self):
        self.logger.info('Start Generating Features for RLIPP Calculation...')
        output_dir = self.config.save_dir
        if not os.path.exists(output_dir):
            self.logger.error('Bad Output Path: %s' % output_dir)
            exit(1)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.data_loader)):
                data, target = batch.to(self.device)
                output = self.model(data)
                feature = self._out2feature_map(output)
                for k, v in feature.items():
                    path = os.path.join(output_dir, k)
                    f = open(path, 'a', encoding='utf-8')
                    f.write(v)

        return None

    def _out2feature_map(self, output: tuple) -> dict:
        """
        param aux_out: [batch_size, 1]
        param feature: [batch_size, vnn_hid]
        """
        res = dict()
        pred, features_map = output
        final = pred['final']
        res['Final.txt'] = self._feature2text(final)
        pred = pred[self.model.vnn.root]
        res['Final_root.txt'] = self._feature2text(pred)
        for term, feature in features_map.items():
            if term.startswith('final') or term.startswith('drug'):
                continue
            res["%s.txt" % term] = self._feature2text(feature)
        return res

    def _feature2text(self, feature: torch.Tensor) -> str:
        """
        param feature: [batch_size, feature_size]
        """
        feature = feature.cpu().numpy().tolist()
        feature_text = ['\t'.join([str(i) for i in x]) for x in feature]
        feature_text = ('\n').join(feature_text) + '\n'
        return feature_text

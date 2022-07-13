import sys
import torch
import logging
import torch.nn as nn

from functools import partial

from model.vnn import VNNModel
from model.drug import DrugModel
from config import cfg2crt


class DrugCellModel(nn.Module):
    def __init__(self, cfg):
        super(NewDrugCellModel, self).__init__()
        self.logger = logging.getLogger('drugCellModel')
        self.vnn = VNNModel(cfg.vnn)
        self.drug = DrugModel(cfg.drug)
        self.criterion = None
        self.init_criterion(cfg.criterion)
        self.construct_final_linear(cfg.final_hid)

    def construct_final_linear(self, final_hidden_size):
        final_linear_input = self.vnn.num_hiddens_genotype + \
            self.drug.num_hiddens_drug[-1]
        self.add_module('final_linear_layer', nn.Linear(
            final_linear_input, final_hidden_size))
        self.add_module('final_batchnorm_layer',
                        nn.BatchNorm1d(final_hidden_size))
        self.add_module('final_aux_linear_layer',
                        nn.Linear(final_hidden_size, 1))
        self.add_module('final_linear_layer_output', nn.Linear(1, 1))

    def init_criterion(self, crt: str):
        criterion = cfg2crt[crt]
        if crt.startswith('text'):
            self.criterion = partial(criterion, lambda_r=0.0, lambda_t=0.1)
        else:
            self.criterion = partial(criterion, lambda_r=0.2)

    def init_by_mask(self):
        self.vnn.init_by_mask()

    def update_by_mask(self):
        self.vnn.update_by_mask()

    def _to_(self, device):
        for key, val in self.vnn.term_mask_map.items():
            self.vnn.term_mask_map[key] = val.to(device)

    def train_step(self, x, y_gold):
        """
        params data: tuple of Tensor [batch], which corresponding to cell id and drug id
        params label: Tensor [batch]
        return loss: Tensor of final loss
        return pred: equal to aux_out_map in forward function
        """
        cell_ids, drug_ids = x
        term_NN_out_map, aux_out_map = dict(), dict()
        term_vnn_out_map, aux_vnn_out_map = self.vnn(cell_ids)
        drug_out, term_drug_out_map, aux_drug_out_map = self.drug(drug_ids)

        term_NN_out_map.update(term_vnn_out_map)
        term_NN_out_map.update(term_drug_out_map)
        aux_out_map.update(aux_vnn_out_map)
        aux_out_map.update(aux_drug_out_map)
        final_input = torch.cat(
            (term_vnn_out_map[self.vnn.root], drug_out), dim=1)
        final_output = self._modules['final_batchnorm_layer'](
            torch.tanh(self._modules['final_linear_layer'](final_input)))
        term_NN_out_map['final'] = final_output
        aux_layer_out = torch.tanh(
            self._modules['final_aux_linear_layer'](final_output))
        aux_out_map['final'] = self._modules['final_linear_layer_output'](
            aux_layer_out)

        if self.crt.startswith('text'):
            for term, term_hidden in term_vnn_out_map.items():
                term_text_embedding = getattr(
                    self.vnn, '%s_text.feature' % term)
                term_text_hidden = self.vnn._modules['%s_text_linear_layer' % term](
                    term_text_embedding)
                term_text_hidden = torch.tanh(term_text_hidden) - term_hidden
                aux_out_map['text_%s' %
                            term] = term_text_hidden.norm(dim=0, p=2)

        loss = self.criterion(aux_out_map, y_gold)
        return loss, aux_out_map['final']

    def forward(self, x):
        """
            x: tuple of Tensor [batch], which corresponding to cell id and drug id
        """
        cell_ids, drug_ids = x
        term_NN_out_map, aux_out_map = dict(), dict()
        aux_vnn_out_map, term_vnn_out_map = self.vnn(cell_ids)
        drug_out, aux_drug_out_map, term_drug_out_map = self.drug(drug_ids)

        term_NN_out_map.update(term_vnn_out_map)
        term_NN_out_map.update(term_drug_out_map)
        aux_out_map.update(aux_vnn_out_map)
        aux_out_map.update(aux_drug_out_map)
        final_input = torch.cat(
            (term_vnn_out_map[self.vnn.root], drug_out), dim=1)
        final_output = self._modules['final_batchnorm_layer'](
            torch.tanh(self._modules['final_linear_layer'](final_input)))
        term_NN_out_map['final'] = final_output
        aux_layer_out = torch.tanh(
            self._modules['final_aux_linear_layer'](final_output))
        aux_out_map['final'] = self._modules['final_linear_layer_output'](
            aux_layer_out)

        return aux_out_map, term_NN_out_map

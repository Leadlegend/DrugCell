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
        super(DrugCellModel, self).__init__()
        self.logger = logging.getLogger('drugCellModel')
        self.vnn = VNNModel(cfg.vnn)
        self.drug = DrugModel(cfg.drug)
        self.crt = cfg.criterion
        self.criterion = None
        self.num_hiddens_final = cfg.final_hid
        self.init_criterion()
        self.construct_final_linear()

    def construct_final_linear(self):
        input_size = self.vnn.num_hiddens_genotype + \
            self.drug.num_hiddens_drug[-1]
        for layer_index in range(len(self.num_hiddens_drug)):
            self.add_module('final_linear_layer_%d' % (layer_index+1),
                            nn.Linear(input_size, self.num_hiddens_drug[layer_index]))
            self.add_module('final_batchnorm_layer_%d' % (layer_index+1),
                            nn.BatchNorm1d(self.num_hiddens_drug[layer_index]))
            input_size = self.num_hiddens_drug[layer_index]

        self.add_module('final_aux_linear_layer1',
                        nn.Linear(self.num_hiddens_drug[-1], 1))
        self.add_module('final_aux_linear_layer2', nn.Linear(1, 1))

    def init_criterion(self):
        criterion = cfg2crt[self.crt.name]
        if self.crt.name.startswith('text'):
            self.criterion = partial(
                criterion, lambda_r=self.crt.lambda_r, lambda_t=self.crt.lambda_t)
        else:
            self.criterion = partial(criterion, lambda_r=self.crt.lambda_r)

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
        aux_vnn_out_map, term_vnn_out_map = self.vnn(cell_ids)
        drug_out, aux_drug_out_map, term_drug_out_map = self.drug(drug_ids)

        term_NN_out_map.update(term_vnn_out_map)
        term_NN_out_map.update(term_drug_out_map)
        #aux_out_map.update(aux_vnn_out_map)
        #aux_out_map.update(aux_drug_out_map)
        final_input = torch.cat((term_vnn_out_map[self.vnn.root], drug_out),
                                dim=1)
        for i in range(1, len(self.num_hiddens_drug)+1, 1):
            final_input = self._modules['final_linear_layer_' +
                                        str(i)](final_input)
            final_input = torch.relu_(final_input)
            final_input = self._modules['final_batchnorm_layer_' +
                                        str(i)](final_input)
        aux_layer1_out = torch.tanh(
            self._modules['final_aux_linear_layer1'](final_input))
        aux_out_map['final'] = self._modules['final_aux_linear_layer2'](
            aux_layer1_out)


        if self.crt.name.startswith('text'):
            for term, term_hidden in term_vnn_out_map.items():
                term_text_embedding = getattr(self.vnn,
                                              '%s_text-feature' % term)
                term_text_hidden = self.vnn._modules['%s_text_linear_layer' %
                                                     term](term_text_embedding)
                term_text_hidden = torch.tanh(term_text_hidden) - term_hidden
                aux_out_map['text_%s' % term] = torch.mean(
                    term_text_hidden.norm(dim=-1, p=2))

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
        final_input = torch.cat((term_vnn_out_map[self.vnn.root], drug_out),
                                dim=1)
        for i in range(1, len(self.num_hiddens_drug)+1, 1):
            final_input = self._modules['final_linear_layer_' +
                                        str(i)](final_input)
            final_input = torch.relu_(final_input)
            final_input = self._modules['final_batchnorm_layer_' +
                                        str(i)](final_input)
        aux_layer1_out = torch.tanh(
            self._modules['final_aux_linear_layer1'](final_input))
        aux_out_map['final'] = self._modules['final_aux_linear_layer2'](aux_layer1_out)


        return aux_out_map, term_NN_out_map

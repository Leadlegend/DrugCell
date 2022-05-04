import os
import sys
import torch
import logging
import numpy as np
import torch.nn as nn


class DrugModel(nn.Module):
    def __init__(self, cfg):
        super(DrugModel, self).__init__()
        self.logger = logging.getLogger('drugModel')
        self.num_hiddens_drug = cfg.drug_hid
        self.fingerprint, self.drug_dim, _ = self.construct_embedding(
            cfg.drug_embed)
        self.construct_NN_drug()

    def construct_embedding(self, embed_path):
        self.logger.info('Constructing Drug Fingerprint...')
        if not os.path.exists(embed_path):
            self.logger.error('Bad Drug Fingerprint File.')
            sys.exit(1)
        feature = np.genfromtxt(embed_path, delimiter=',')
        embedding_weights = torch.from_numpy(feature)
        embed_size, dim_drug = embedding_weights.shape
        embedding = nn.Embedding(
            num_embeddings=embed_size,
            embedding_dim=dim_drug,
            padding_idx=None).from_pretrained(
            embeddings=embedding_weights,
            freeze=True
        )
        return embedding, dim_drug, embed_size

    # add modules for fully connected neural networks for drug processing
    def construct_NN_drug(self):
        self.logger.info('Constructing Drug MLP...')
        input_size = self.drug_dim
        for layer_index in range(len(self.num_hiddens_drug)):
            self.add_module('drug_linear_layer_' + str(layer_index+1),
                            nn.Linear(input_size, self.num_hiddens_drug[layer_index]))
            self.add_module('drug_batchnorm_layer_' + str(layer_index+1),
                            nn.BatchNorm1d(self.num_hiddens_drug[layer_index]))
            self.add_module('drug_aux_linear_layer1_' + str(layer_index+1),
                            nn.Linear(self.num_hiddens_drug[layer_index], 1))
            self.add_module('drug_aux_linear_layer2_' +
                            str(layer_index+1), nn.Linear(1, 1))
            input_size = self.num_hiddens_drug[layer_index]

    def forward(self, drug_input):
        """
            drug_input: [batch_size]
        """
        term_NN_out_map = dict()
        aux_out_map = dict()
        drug_out = self.fingerprint(drug_input).float()
        # [batch_size, gene_num]
        for i in range(1, len(self.num_hiddens_drug)+1, 1):
            drug_out = self._modules['drug_batchnorm_layer_'+str(i)](
                torch.tanh(self._modules['drug_linear_layer_' + str(i)](drug_out)))
            term_NN_out_map['drug_'+str(i)] = drug_out

            #self.logger.debug(drug_out.shape)
            aux_layer1_out = torch.tanh(
                self._modules['drug_aux_linear_layer1_'+str(i)](drug_out))
            aux_out_map['drug_' +
                        str(i)] = self._modules['drug_aux_linear_layer2_'+str(i)](aux_layer1_out)

        return drug_out, term_NN_out_map, aux_out_map

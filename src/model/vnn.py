import os
import sys
import torch
import logging
import numpy as np
import networkx as nx
import torch.nn as nn
import networkx.algorithms.dag as nxadag
import networkx.algorithms.components.connected as nxacc

from tqdm import tqdm
from data.tokenizer import load_vocab


class VNNModel(nn.Module):
    def __init__(self, cfg):
        super(VNNModel, self).__init__()
        self.num_hiddens_genotype = cfg.gene_hid
        self.num_hiddens_text = cfg.text_dim if cfg.text_dim is not None else 0
        self.term_dim_map = dict()
        self.term_mask_map = dict()
        # term_neighbor_map records all children of each term
        self.term_neighbor_map = dict()
        # term_layer_list stores the built neural network
        self.term_layers = list()
        self.logger = logging.getLogger('vnnModel')
        self.embedding, self.gene_dim, _ = self.construct_embedding(
            cfg.cell_embed)

        dG, self.root, term_size_map, self.term_direct_gene_map = self.load_ontology(
            cfg.onto, cfg.gene2idx)
        self.cal_term_dim(term_size_map)
        self.cal_term_mask()
        if self.num_hiddens_text > 0:
            self.construct_text_embedding(dG)

        self.contruct_direct_gene_layer()
        self.construct_vnn(dG)

    def load_ontology(self, onto_path, gene2idx_file):
        self.logger.info("Parsing Featured-Gene Set...")
        if not os.path.exists(gene2idx_file):
            self.logger.error("Bad Gene2idx File %s" % gene2idx_file)
            sys.exit(1)
        gene2idx = load_vocab(gene2idx_file, has_index=True)
        self.logger.info("Parsing Ontology Structure...")
        gene_set = set()
        dG = nx.DiGraph()
        term_size_map = dict()
        term_direct_gene_map = dict()
        if not os.path.exists(onto_path):
            self.logger.error('Bad Ontology Path.')
            sys.exit(1)
        handle = open(onto_path)
        for line in tqdm(handle.readlines()):
            line = line.rstrip().split('\t')
            if line[2] == 'default':
                # add edge between subsystems
                dG.add_edge(line[0], line[1])
            else:
                if line[1] not in gene2idx:
                    continue
                if line[0] not in term_direct_gene_map:
                    term_direct_gene_map[line[0]] = set()
                term_direct_gene_map[line[0]].add(gene2idx[line[1]])
                gene_set.add(line[1])
                # add edges between subsystem and gene
                # where line0 is subsys and line1 is gene
        handle.close()

        for term in dG.nodes():
            term_gene_set = set()
            if term in term_direct_gene_map:
                term_gene_set = term_direct_gene_map[term]
            deslist = nxadag.descendants(dG, term)
            for child in deslist:
                if child in term_direct_gene_map:
                    term_gene_set = term_gene_set | term_direct_gene_map[child]
            # calculate the number of gene relating with every term
            if len(term_gene_set) == 0:
                self.logger.error('There is an empty term: %s' % term)
                sys.exit(1)
            else:
                term_size_map[term] = len(term_gene_set)

        leaves = [n for n in dG.nodes if dG.in_degree(n) == 0]
        uG = dG.to_undirected()
        connected_subG_list = list(nxacc.connected_components(uG))

        self.logger.info('There are %d genes' % len(gene_set))
        self.logger.info('There are %d terms' % len(dG.nodes()))
        self.logger.info('There are %d connected componenets' %
                         len(connected_subG_list))
        self.logger.info('There are %d roots: %s' % (len(leaves), leaves[0]))
        if len(leaves) > 1:
            self.logger.error(
                'There are more than 1 root of ontology. Please use only one root.')
            sys.exit(1)
        if len(connected_subG_list) > 1:
            self.logger.error(
                'There are more than connected components. Please connect them.')
            sys.exit(1)
        return dG, leaves[0], term_size_map, term_direct_gene_map

    def construct_embedding(self, embed_path):
        """
        return embedding: cell-line id -> gene mutation vector (size: num_gene)
        """
        self.logger.info('Constructing Cell Mutational Feature...')
        if not os.path.exists(embed_path):
            self.logger.error('Bad Mutation File.')
            sys.exit(1)
        feature = np.genfromtxt(embed_path, delimiter=',', dtype=np.float32)
        embedding_weights = torch.from_numpy(feature)
        embed_size, num_gene = embedding_weights.shape
        embedding = nn.Embedding(
            num_embeddings=embed_size,
            embedding_dim=num_gene,
            padding_idx=None).from_pretrained(
            embeddings=embedding_weights,
            freeze=True
        )
        return embedding, num_gene, embed_size

    def contruct_direct_gene_layer(self):
        self.logger.info('Constructing Gene Layers...')
        for term, gene_set in self.term_direct_gene_map.items():
            if len(gene_set) == 0:
                self.logger.error(
                    'There are no directed asscoiated genes for %s' % term)
                sys.exit(1)
            # if there are some genes directly annotated with the term
            # add a layer taking in all genes and forwarding out only those genes
            self.add_module(term+'_direct_gene_layer',
                            nn.Linear(self.gene_dim, len(gene_set)))

    # start from bottom (leaves), and start building a neural network using the given ontology
    # adding modules --- the modules are not connected yet
    def construct_vnn(self, dG):
        self.logger.debug("Constructing Term-Neighbor Mapping.")
        for term in dG.nodes():
            self.term_neighbor_map[term] = []
            for child in dG.neighbors(term):
                self.term_neighbor_map[term].append(child)
        self.logger.info(
            'Constructing VNN Network Based on Ontology Knowledge.')
        while True:
            leaves = [n for n in dG.nodes() if dG.out_degree(n) == 0]
            if len(leaves) == 0:
                break
            # notice: here the author used out_degree
            # while preprocessing, the author used in_degree to find the root node
            self.term_layers.append(leaves)
            for term in leaves:
                # input size will be #chilren + #genes directly annotated by the term
                input_size = 0
                for child in self.term_neighbor_map[term]:
                    input_size += self.term_dim_map[child]
                if term in self.term_direct_gene_map:
                    input_size += len(self.term_direct_gene_map[term])
                # term_hidden is the number of the hidden variables in each state
                term_hidden = self.term_dim_map[term]
                self.add_module(term+'_linear_layer',
                                nn.Linear(input_size, term_hidden))
                self.add_module(term+'_batchnorm_layer',
                                nn.BatchNorm1d(term_hidden))
                self.add_module(term+'_aux_linear_layer1',
                                nn.Linear(term_hidden, 1))
                self.add_module(term+'_aux_linear_layer2', nn.Linear(1, 1))
            dG.remove_nodes_from(leaves)

    def construct_text_embedding(self, dG):
        self.logger.info('Initialize torch Buffer to store text features of GO Term in VNN' )

        terms = dG.nodes()
        for term in terms:
            term_text_data = torch.zeros(size=[self.num_hiddens_text])
            self.register_buffer('%s_text-feature'% term, term_text_data)
            self.add_module('%s_text_linear_layer'% term, nn.Linear(self.num_hiddens_text, self.num_hiddens_genotype))

        self.logger.info('Finished buffer registration for %d terms' % len(terms))

    def cal_term_dim(self, term_size_map):
        """
        Trivial(uniform) Distribution for neuron_num of each GO Term
        """
        for term, term_size in term_size_map.items():
            num_output = self.num_hiddens_genotype
            self.logger.debug("term\t%s\tterm_size\t%d\tnum_hiddens\t%d" %
                              (term, term_size, num_output))
            self.term_dim_map[term] = num_output

    def cal_term_mask(self):
        """
        Calculate the connection mask between each Term and Gene in VNN
        which will ensure each GO Term in VNN only have access to their relevant Gene
        term_mask_map: dict[term_name] -> mask
        where mask: (num_direct-gene_of_term, num_gene)
        """
        for term, gene_set in self.term_direct_gene_map.items():
            mask = torch.zeros(len(gene_set), self.gene_dim)
            for i, gene_id in enumerate(gene_set):
                mask[i, gene_id] = 1
            self.term_mask_map[term] = mask
        return self.term_mask_map

    def init_by_mask(self):
        """
        This function guarantee that Linear Layer between GO Terms and thier relevant Genes is shaped by VNN Structure
        """
        if len(self.term_dim_map):
            for name, param in self.named_parameters():
                if name.endswith('_direct_gene_layer.weight'):
                    term = name.split('_')[0]
                    param.mul_(self.term_mask_map[term])

    def update_by_mask(self):
        """
        Given that every Term in VNN has access to all of Gene
        While actually, we want Term to follow prior knowledge of GO 
        which was defined by direct Gene map or term_mask_map
        """
        for name, param in self.named_parameters():
            if '_direct_gene_layer.weight' not in name:
                continue
            term = name.split('_')[0]
            param.grad.data = torch.mul(
                param.grad.data, self.term_mask_map[term])

    def forward(self, gene_input):
        """
        params: gene_input: [batch_size]
        return: term_NN_out_map: dict of Neural output of every term in VNN
        return: aux_out_map: dict of predicted result of every term in VNN
        """
        term_gene_out_map = dict()
        term_NN_out_map = dict()
        aux_out_map = dict()
        gene_embedded = self.embedding(gene_input)
        # (batch_size, enbed_size)

        for term in self.term_direct_gene_map.keys():
            term_gene_out_map[term] = self._modules[term +
                                                    '_direct_gene_layer'](gene_embedded)
            # For each term,

        for i, layer in enumerate(self.term_layers):
            for term in layer:
                child_input_list = list()
                for child in self.term_neighbor_map[term]:
                    child_input_list.append(term_NN_out_map[child])
                if term in self.term_direct_gene_map:
                    child_input_list.append(term_gene_out_map[term])

                child_input = torch.cat(child_input_list, dim=1)
                # concat the output of its children as input

                term_linear_out = self._modules[term +
                                                '_linear_layer'](child_input)

                term_Tanh_out = torch.tanh(term_linear_out)
                term_NN_out_map[term] = self._modules[term +
                                                      '_batchnorm_layer'](term_Tanh_out)
                # self.logger.debug(term_NN_out_map[term].shape)
                aux_layer1_out = torch.tanh(
                    self._modules[term+'_aux_linear_layer1'](term_NN_out_map[term]))
                aux_out_map[term] = self._modules[term +
                                                  '_aux_linear_layer2'](aux_layer1_out)
        return aux_out_map, term_NN_out_map

import sys
import torch
import argparse
import numpy as np
import networkx as nx
import networkx.algorithms.dag as nxadag
import networkx.algorithms.components.connected as nxacc


def args_util():
    # load the params for training using argparse
    parser = argparse.ArgumentParser(description='Train dcell')
    parser.add_argument(
        '-onto', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('-train', help='Training dataset', type=str)
    parser.add_argument('-val', help='Validation dataset', type=str)
    parser.add_argument(
        '-epoch', help='Training epochs for training', type=int, default=300)
    parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
    parser.add_argument('-optimizer', help="Optimizor",
                        type=str, default="adam")
    parser.add_argument(
        '-sched', help="Learning rate scheduler", type=str, default="None")
    parser.add_argument('-batch', help='Batchsize', type=int, default=5000)
    parser.add_argument(
        '-ckpt', help='Path of model checkpoint', type=str, default=None)
    parser.add_argument(
        '-save_dir', help='Folder for trained models', type=str, default='./model/')
    parser.add_argument('-device', help='Specify GPU', type=int, default=0)
    parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str)
    parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str)
    parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str)

    parser.add_argument(
        '-geno_hid', help='Mapping for the number of neurons in each term in genotype parts', type=int, default=6)
    parser.add_argument(
        '-drug_hid', help='Mapping for the number of neurons in each layer', type=str, default='100,50,6')
    parser.add_argument(
        '-final_hid', help='The number of neurons in the top layer', type=int, default=6)

    parser.add_argument(
        '-cell_embed', help='Mutation information for cell lines', type=str)
    parser.add_argument(
        '-drug_embed', help='Morgan fingerprint representation for drugs', type=str)

    return parser.parse_args()


def pearson_corr(x, y):
    xx = x - torch.mean(x)
    yy = y - torch.mean(y)

    return torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy, 2))


def load_ontology(file_name, gene2id_mapping):

    dG = nx.DiGraph()
    term_direct_gene_map = {}
    term_size_map = {}

    file_handle = open(file_name)

    gene_set = set()

    for line in file_handle:

        line = line.rstrip().split()

        if line[2] == 'default':
            # add edge between subsystems
            dG.add_edge(line[0], line[1])
        else:
            # add edges between subsystem and gene
            # where line0 is subsys and line1 is gene
            if line[1] not in gene2id_mapping:
                continue

            if line[0] not in term_direct_gene_map:
                term_direct_gene_map[line[0]] = set()

            term_direct_gene_map[line[0]].add(gene2id_mapping[line[1]])

            gene_set.add(line[1])

    file_handle.close()

    print('There are', len(gene_set), 'genes')

    for term in dG.nodes():
        # calculate the number of gene relating with every term
        term_gene_set = set()

        if term in term_direct_gene_map:
            term_gene_set = term_direct_gene_map[term]

        deslist = nxadag.descendants(dG, term)

        for child in deslist:
            if child in term_direct_gene_map:
                term_gene_set = term_gene_set | term_direct_gene_map[child]

        # jisoo
        if len(term_gene_set) == 0:
            print('There is empty terms, please delete term:', term)
            sys.exit(1)
        else:
            term_size_map[term] = len(term_gene_set)

    leaves = [n for n in dG.nodes if dG.in_degree(n) == 0]
    #leaves = [n for n,d in dG.in_degree() if d==0]

    uG = dG.to_undirected()
    connected_subG_list = list(nxacc.connected_components(uG))

    print('There are', len(leaves), 'roots:', leaves[0])
    print('There are', len(dG.nodes()), 'terms')
    print('There are', len(connected_subG_list), 'connected componenets')

    if len(leaves) > 1:
        print('There are more than 1 root of ontology. Please use only one root.')
        sys.exit(1)
    if len(connected_subG_list) > 1:
        print('There are more than connected components. Please connect them.')
        sys.exit(1)

    return dG, leaves[0], term_size_map, term_direct_gene_map


def load_train_data(file_name, cell2id, drug2id):
    feature = []
    label = []

    with open(file_name, 'r') as fi:
        for line in fi:
            tokens = line.strip().split('\t')

            feature.append([cell2id[tokens[0]], drug2id[tokens[1]]])
            label.append([float(tokens[2])])

    return feature, label


def prepare_predict_data(test_file, cell2id_mapping_file, drug2id_mapping_file):

    # load mapping files
    cell2id_mapping = load_mapping(cell2id_mapping_file)
    drug2id_mapping = load_mapping(drug2id_mapping_file)

    test_feature, test_label = load_train_data(
        test_file, cell2id_mapping, drug2id_mapping)

    print('Total number of cell lines = %d' % len(cell2id_mapping))
    print('Total number of drugs = %d' % len(drug2id_mapping))

    return (torch.Tensor(test_feature), torch.Tensor(test_label)), cell2id_mapping, drug2id_mapping


def load_mapping(mapping_file):

    mapping = {}

    file_handle = open(mapping_file)

    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])

    file_handle.close()

    return mapping


def prepare_train_data(train_file, test_file, cell2id_mapping_file, drug2id_mapping_file):

    # load mapping files
    cell2id_mapping = load_mapping(cell2id_mapping_file)
    drug2id_mapping = load_mapping(drug2id_mapping_file)

    train_feature, train_label = load_train_data(
        train_file, cell2id_mapping, drug2id_mapping)
    test_feature, test_label = load_train_data(
        test_file, cell2id_mapping, drug2id_mapping)

    print('Total number of cell lines = %d' % len(cell2id_mapping))
    print('Total number of drugs = %d' % len(drug2id_mapping))

    return (torch.Tensor(train_feature), torch.FloatTensor(train_label), torch.Tensor(test_feature), torch.FloatTensor(test_label)), cell2id_mapping, drug2id_mapping


def build_input_vector(input_data, cell_features, drug_features):
    genedim = len(cell_features[0, :])
    drugdim = len(drug_features[0, :])
    feature = np.zeros((input_data.size()[0], (genedim+drugdim)))

    for i in range(input_data.size()[0]):
        feature[i] = np.concatenate((cell_features[int(
            input_data[i, 0])], drug_features[int(input_data[i, 1])]), axis=None)

    feature = torch.from_numpy(feature).float()
    return feature

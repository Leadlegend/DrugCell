import json
import networkx as nx
import networkx.algorithms.dag as nxadag
import networkx.algorithms.components.connected as nxacc
from yaml import load

from config import *


def get_terms():
    f = open(vnn_stats_path, 'r', encoding='utf-8')
    g = open(vnn_info_path, 'r', encoding='utf-8')
    name2num = dict()
    reserved_terms = list()
    name2id = json.load(g)
    for line in f:
        name, num = line.strip().split('\t')
        num = int(num)
        if num >= 10: 
            name2num[name] = num
    for name in name2num.keys():
        reserved_terms.extend(name2id[name])

    print("There are %d terms to be reserved." %len(reserved_terms))
    print('GO:0008150' in reserved_terms)
    h = open(vnn_new_path, 'w', encoding='utf-8')
    h.write('\n'.join(reserved_terms))
    return reserved_terms


def load_ontology(file_name):
    dG = nx.DiGraph()
    file_handle = open(file_name)

    for line in file_handle:
        line = line.rstrip().split()
        if line[2] == 'default':
            dG.add_edge(line[0], line[1])
        else:
            continue

    file_handle.close()

    leaves = [n for n in dG.nodes if dG.in_degree(n) == 0]
    uG = dG.to_undirected()
    connected_subG_list = list(nxacc.connected_components(uG))

    print('There are', len(leaves), 'roots:', leaves[0])
    print('There are', len(dG.nodes()), 'terms')
    print('There are', len(connected_subG_list), 'connected componenets')
    if len(leaves) > 1:
        print('There are more than 1 root of ontology. Please use only one root.')
        print(leaves)
    if len(connected_subG_list) > 1:
        print('There are more than connected components. Please connect them.')

    return dG, leaves[0]


def get_dg_info(dG, root):
    layers = 0
    print(dG.out_degree(root))
    print(dG['GO:0051641'])
    print(dG.pred['GO:0023058'])
    print(dG.succ['GO:0051641'])
    print(dG.adj['GO:0051641'])
    while True:
        leaves = [n for n in dG.nodes() if dG.out_degree(n) == 0]
        if len(leaves) == 0:
            break
        layers += 1
        dG.remove_nodes_from(leaves)
        print(len(leaves))


def prune(dG: nx.DiGraph, terms, root):
    dg_copy = nx.DiGraph(dG)
    leaves, last_leaves = [], []
    layer = 0
    for i in range(5):
        leaves = [n for n in dG.nodes() if dG.out_degree(n) == 0]
        layer_num, del_num = len(leaves), 0
        for leaf in leaves:
            if leaf not in terms:
                del_num += 1
                preds = dg_copy.pred[leaf].keys()
                succs = dg_copy.succ[leaf].keys()
                for pred in preds:
                    for succ in succs:
                        dg_copy.add_edge(pred, succ)
                dg_copy.remove_node(leaf)
            dG.remove_nodes_from(leaves)
        print("Delete %d in %d terms of layer %d" %(del_num, layer_num, i))
    print("%d Terms left" %len(dg_copy.nodes()))
    leaves = [n for n in dg_copy.nodes if dg_copy.in_degree(n) == 0]
    print('There are', len(leaves), 'roots:', leaves[0])
    return dg_copy


def graph2txt(dg: nx.DiGraph, terms, old_path, new_path):
    f = open(old_path, 'r', encoding='utf-8')
    g = open(new_path, 'w', encoding='utf-8')
    edges = [[e[0], e[1], 'default'] for e in dg.edges()]
    edges = ['\t'.join(e) for e in edges]
    def_num, gen_num = len(edges), 0
    res_default = '\n'.join(edges) + '\n'
    g.write(res_default)
    for line in f:
        data = line.strip().split('\t')
        assert len(data) == 3
        if data[-1] == 'gene' and (data[0] in terms or data[1] in terms):
            g.write(line)
            gen_num += 1
    print("write %d default lines and %d gene lines into new ontology file." %(def_num, gen_num))


def main():
    terms = get_terms()
    dG, root = load_ontology(dg_path)
    #get_dg_info(dG, root)
    pruned_dg = prune(dG, terms, root)
    graph2txt(pruned_dg, terms, dg_path, dg_new_path)
    


if __name__ == '__main__':
    #main()
    get_terms()
    load_ontology(dg_new_path)


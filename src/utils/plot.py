import os
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import stats
from copy import deepcopy
from typing import Optional, List, Union
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
mpl.use('tkagg')


def get_rlipp_vectors(rlipp_file, cell2id, terms):
    if not os.path.exists(rlipp_file):
        exit(1)
    cell_index = pd.read_csv(
            cell2id, sep="\t", header=None, names=['Index', 'Cell'], dtype={'Index': int, 'Cell': str})
    cell2index = dict(
        zip(cell_index['Cell'], cell_index['Index']))
    #cell2index = dict(zip(cells, range(len(cells))))
    term2id = {t: i for i, t in enumerate(terms)}
    rlipp_data = pd.read_csv(rlipp_file, encoding='utf-8',
                             sep='\t', header=0, usecols=[0, 1, 6])
    rlipp_vector = np.zeros(
        [len(terms), len(cell2index)], dtype=np.float64)

    for i, row in tqdm(rlipp_data.iterrows()):
        term, index, rlipp = row[0], row[1], row[2]
        if term not in terms:
            continue
        assert index in cell2index
        x = term2id[term]
        y = cell2index[index]
        rlipp_vector[x, y] = rlipp
    return rlipp_vector


def get_text_vectors(text_embed_path, terms):
    ckpt = torch.load(text_embed_path, map_location='cpu')
    text_vector = np.zeros([len(terms), 768], dtype=np.float64)
    for x, term in enumerate(terms):
        key = "vnn.%s_text-feature" % term
        text_vector[x] = ckpt[key].detach().numpy()
    return text_vector


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * \
        np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return (0.5 + 0.5 * res).flatten()


def get_vectors_similarity(vectors):
    v2 = deepcopy(vectors)
    xs = get_cos_similar_matrix(vectors, v2)
    return xs


def get_random_data(size=10000):
    # Generate fake data
    x = np.random.normal(size=size)
    y = x * 3 + np.random.normal(size=size)
    return x, y


def scatter_plot_with_density_kde(xs: List[float], ys: List[float]):
    # Calculate the point density
    xy = np.vstack([xs, ys])
    z = gaussian_kde(xy)(xy)
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, c=z, s=100, edgecolor='')
    plt.show()


def scatter_plot_with_density_hist2d(xs, ys, ver):
    cor, _ = stats.spearmanr(xs, ys)
    print(cor)
    plt.hist2d(xs, ys, bins=1000, norm=LogNorm())
    plt.colorbar()
    linear_model = np.polyfit(xs, ys, 1)
    print(linear_model)
    linear_model_fn = np.poly1d(linear_model)
    x = np.arange(0, 2)
    plt.plot(x, linear_model_fn(x), color='red')
    plt.text(x=0.95, y=0.95, s='y=%.3fx+%.3f' % (linear_model[0], linear_model[1]), color='red')
    plt.xlabel("RLIPP Similarity")
    plt.ylabel('Text Embedding Similarity')
    plt.title("ver.%s: spearman correlation: %.3f" % (ver, cor))
    plt.savefig('./text_ver%s.png' % ver)
    plt.show()


def main(index: Union[int, list] = 1):
    rlipp_path = 'data/rlipp.txt'
    cell2id = 'data/cell2ind.txt'
    term_paths = ['data/go_text/vnn.txt',
                  'data/go_text/vnn_non-zero.txt', 'data/go_text/vnn_small.txt']

    if isinstance(index, int):
        index = [index]
    for i in index:
        term_path = term_paths[i-1]
        term_embed_path = 'ckpt/text/dc_text_v%d.pt' % (i)
        terms = [line.strip()
                 for line in open(term_path, 'r', encoding='utf-8')]
        rlipp_vectors = get_rlipp_vectors(rlipp_path, cell2id, terms)
        text_vectors = get_text_vectors(term_embed_path, terms)
        print("text vectors:", text_vectors.shape)
        print("rlipp vectors:", rlipp_vectors.shape)
        xs, ys = get_vectors_similarity(
            rlipp_vectors), get_vectors_similarity(text_vectors)
        scatter_plot_with_density_hist2d(xs, ys, ver=i)


if __name__ == '__main__':
    main([1, 2, 3])

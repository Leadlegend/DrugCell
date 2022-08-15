import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional, List
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
from copy import deepcopy
mpl.use('tkagg')


def get_rlipp_vectors(rlipp_file, terms):
    if not os.path.exists(rlipp_file):
        exit(1)
    cell2index = dict(
        zip(self.cell_index['Cell'], self.cell_index['Index']))
    term2id = {t: i for i, t in enumerate(terms)}
    rlipp_data = pd.read_csv(rlipp_file, encoding='utf-8',
                                 sep='\t', header=0, use_cols=[0, 1, 6])
    rlipp_vector = np.zeros(
        [len(terms), len(cell2index)], dtype=float64)

    for i, row in tqdm(rlipp_data.iterrows()):
        term, index, rlipp = row[0], row[1], row[2]
        if i < 10:
            print("%s\t%s\t%s" % (term, index, rlipp))
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
        key = "vnn.%s_text-feature" %term
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


def scatter_plot_with_density_hist2d(xs, ys):
    plt.hist2d(xs, ys, bins=1000, norm=LogNorm())
    plt.colorbar()
    linear_model = np.polyfit(xs, ys, 1)
    print(linear_model)
    linear_model_fn = np.poly1d(linear_model)
    x = np.arange(0, 2)
    plt.plot(x, linear_model_fn(x), color='red')
    plt.show()


def main():
    term_embed_path = 'ckpt/text/dc_text_v2.pt'
    term_path = 'data/go_text/vnn_non-zero.txt'
    rlipp_path = 'data/rlipp.txt'
    terms = [line.strip() for line in open(term_path, 'r', encoding='utf-8')]
    #rlipp_vectors = get_rlipp_vectors(rlipp_path, terms)
    rlipp_vectors, _ = get_random_data(size=(len(terms), 2))
    text_vectors = get_text_vectors(term_embed_path, terms)
    print(text_vectors.shape)
    print(rlipp_vectors.shape)
    xs, ys = get_vectors_similarity(rlipp_vectors), get_vectors_similarity(text_vectors)
    print(xs.shape)
    scatter_plot_with_density_hist2d(xs, ys)


if __name__ == '__main__':
    main()

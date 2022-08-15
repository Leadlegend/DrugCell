import os
import time
import torch
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy import stats
from multiprocessing import Pool
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


class RLIPPCalculator():
    def __init__(self, args):
        self.logger = logging.getLogger('RLIPP Calculator')
        self.construct_data(args)

        self.cpu_count = args.cpu_num
        self.hidden_dir = args.feature_dir
        self.rlipp_file = args.rlipp_output
        self.gene_rho_file = args.gene_output
        self.num_hiddens_genotype = args.hid_num
        self.index_flag = args.index_flag if args.index_flag else 'Cell'
        assert self.index_flag in ['Cell', 'Drug']
        self.indexes = list(set(self.dataset[self.index_flag]))
        self.index_count = args.index_num if args.index_num else len(
            self.indexes)
        self.construct_index_position_map()

    def construct_data(self, args):
        self.logger.info('Constructing Dataset...')
        self.ontology = pd.read_csv(args.onto, sep='\t', header=None, names=[
                                    'S', 'T', 'I'], dtype={0: str, 1: str, 2: str})
        self.terms = [line.strip()
                      for line in open(args.terms, 'r', encoding='utf-8')]
        self.dataset = pd.read_csv(
            args.data, sep='\t', header=None, names=['Cell', 'Drug', 'AUC'])
        self.genes = pd.read_csv(args.gene2id, sep='\t',
                                 header=None, names=['Index', 'Gene'])['Gene']
        self.cell_index = pd.read_csv(
            args.cell2id, sep="\t", header=None, names=['Index', 'Cell'], dtype={'Index': int, 'Cell': str})

        self.predicted_vals = np.loadtxt(args.pred)
        #self.predicted_vals = torch.load(args.predicted, map_location='cpu')

    def construct_index_position_map(self):
        # Create a map of a list of the position of specific drug in the dataset file
        self.logger.info('Parsing Dataset...')
        self.index_position_map = {d: [] for d in self.indexes}
        for i, row in tqdm(self.dataset.iterrows()):
            self.index_position_map[row[self.index_flag]].append(i)
        return self.index_position_map

    def create_index_corr_map_sorted(self):
        # Create a sorted map of spearman correlation values for every drug
        drug_corr_map = {}
        for index in self.indexes:
            if len(self.index_position_map[index]) == 0:
                drug_corr_map[index] = 0.0
                continue
            true_vals = np.take(
                np.array(self.dataset['AUC']), self.index_position_map[index])
            pred_vals = self.get_predict_vals_by_index(index)
            drug_corr_map[index] = stats.spearmanr(true_vals, pred_vals)[0]
        return {drug: corr for drug, corr in sorted(drug_corr_map.items(), key=lambda item: item[1], reverse=True)}

    def _load_feature(self, element, size):
        # Load the hidden file for a given element
        file_name = os.path.join(self.hidden_dir, element + '.txt')
        return np.loadtxt(file_name, usecols=range(size))

    def load_term_features(self, term):
        return self._load_feature(term, self.num_hiddens_genotype)

    def load_gene_features(self, gene):
        return self._load_feature(gene, 1)

    # Load hidden features for all the terms and genes
    def load_features(self):
        self.logger.info('Loading features for all the terms...')
        feature_map = {}
        with Pool(self.cpu_count) as p:
            results = p.map(self.load_term_features, self.terms)
        for i, term in enumerate(self.terms):
            feature_map[term] = results[i]
        if self.gene_rho_file is not None:
            self.logger.info('Loading features for all the genes...')
            with Pool(self.cpu_count) as p:
                results = p.map(self.load_gene_features, self.genes)
            for i, gene in enumerate(self.genes):
                feature_map[gene] = results[i]

        self.logger.info('Constructing feature map for children of terms')
        child_feature_map = {t: [] for t in self.terms}
        for term in tqdm(self.terms):
            children = [row[1]
                        for row in self.ontology.itertuples() if row[1] == term and row[2] == 'default']
            for child in children:
                child_feature_map[term].append(feature_map[child])

        return feature_map, child_feature_map

    def get_predict_vals_by_positions(self, positions):
        return np.take(self.predicted_vals, positions)

    def get_predict_vals_by_index(self, index):
        return np.take(self.predicted_vals, self.index_position_map[index])

    # Get a hidden feature matrix of a given term's children
    def get_child_features(self, term_child_features, position_map):
        child_features = []
        for f in term_child_features:
            child_features.append(np.take(f, position_map, axis=0))
        if len(child_features) > 0:
            return np.column_stack([f for f in child_features])
        else:
            return None

    # Executes 5-fold cross validated Ridge regression for a given hidden features matrix
    # and returns the spearman correlation value of the predicted output
    def exec_lm(self, X, y):
        """
        param X: [data_size, feature_size]
        param y: [data_size,]
        return rho: spearman correlation value
        return pred
        """
        pca = PCA(n_components=self.num_hiddens_genotype)
        X_pca = pca.fit_transform(X)
        # X_pca = X
        regr = RidgeCV(cv=5)
        regr.fit(X_pca, y)
        y_pred = regr.predict(X_pca)
        return stats.spearmanr(y_pred, y)

    # Calculates RLIPP for a given term and drug
    # Executes parallely
    def calc_term_rlipp(self, term_features, term_child_features, term, index):
        X_parent = np.take(
            term_features, self.index_position_map[index], axis=0)
        X_child = self.get_child_features(
            term_child_features, self.index_position_map[index])
        y = self.get_predict_vals_by_index(index)
        p_rho, p_pred = self.exec_lm(X_parent, y)
        if X_child is not None:
            c_rho, c_pred = self.exec_lm(X_child, y)
        else:
            c_rho, c_pred = 0, 0
        rlipp = p_rho - c_rho
        result = '{}\t{}\t{:.3e}\t{:.3e}\t{:.3e}\t{:.3e}\t{:.3e}\n'.format(
            term, index, p_rho, p_pred, c_rho, c_pred, rlipp)
        return result

    # Calculates Spearman correlation between Gene embeddings and Predicted AUC
    def calc_gene_rho(self, gene_features, gene, index):
        pred = self.get_predict_vals_by_index(index)
        gene_embeddings = np.take(
            gene_features, self.index_position_map[index])
        rho, p_val = stats.spearmanr(pred, gene_embeddings)
        result = '{}\t{:.3e}\t{:.3e}\n'.format(gene, rho, p_val)
        return result

    # Calculates RLIPP scores for top n drugs (n = drug_count), and
    # prints the result in "Drug Term P_rho C_rho RLIPP" format
    def calc_scores(self):
        self.logger.info('Starting Calculation')
        sorted_indexes = list(self.create_index_corr_map_sorted().keys())[
            0:self.index_count]

        start = time.time()
        feature_map, child_feature_map = self.load_features()
        self.logger.info(
            'Time taken to load features: {:.4f}'.format(time.time() - start))

        rlipp_file, gene_rho_file = None, None
        if self.rlipp_file is not None:
            rlipp_file = open(self.rlipp_file, "w", encoding='utf-8')
            rlipp_file.write(
                    'Term\tIndex\tP_rho\tP_pred\tC_rho\tC_pred\tRLIPP\n')
        if self.gene_rho_file is not None:
            gene_rho_file = open(self.gene_rho_file, "w", encoding='utf-8')
            gene_rho_file.write('Gene\tRho\tP_val\n')

        with Parallel(backend="multiprocessing", n_jobs=self.cpu_count) as parallel:
            for i, index in enumerate(sorted_indexes):
                start = time.time()
                if rlipp_file is not None:
                    rlipp_results = parallel(delayed(self.calc_term_rlipp)(
                        feature_map[term], child_feature_map[term], term, index) for term in self.terms)
                    for result in rlipp_results:
                        rlipp_file.write(result)
                if gene_rho_file is not None:
                    gene_rho_results = parallel(delayed(self.calc_gene_rho)(
                        feature_map[gene], gene, index) for gene in self.genes)
                    for result in gene_rho_results:
                        gene_rho_file.write(result)

                self.logger.info('Index {} completed in {:.4f} seconds'.format(
                    (i+1), (time.time() - start)))

        if gene_rho_file is not None:
            gene_rho_file.close()
        if rlipp_file is not None:
            rlipp_file.close()

    def parse_rlipp_results(self, terms=None):
        if not os.path.exists(self.rlipp_file):
            self.logger.error('Rlipp Result Not Found.')
            exit(1)
        self.cell_index = dict(
            zip(self.cell_index['Cell'], self.cell_index['Index']))
        if terms is None:
            terms = self.terms
        term2id = {t: i for i, t in enumerate(terms)}
        rlipp_data = pd.read_csv(self.rlipp_file, encoding='utf-8',
                                 sep='\t', header=0, use_cols=[0, 1, 6])
        rlipp_vector = np.zeros(
            [len(terms), self.index_count], dtype=np.float64)

        for i, row in tqdm(rlipp_data.iterrows()):
            term, index, rlipp = row[0], row[1], row[2]
            if i < 10:
                print("%s\t%s\t%s" % (term, index, rlipp))
            if term not in terms:
                continue
            assert index in self.cell_index
            x = term2id[term]
            y = self.cell_index[index]
            rlipp_vector[x, y] = rlipp

        return rlipp_vector

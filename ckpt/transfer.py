import torch
import pickle

from tqdm import tqdm
from collections import OrderedDict, defaultdict


def transfer(path, new_path='./model.pt', if_model=False):
    """
        transfer the ckpt from author to ckpt that
    """
    if not if_model:
        sd_old = torch.load(path)
        keys = list(sd_old.keys())
        for key in keys:
            if key.startswith('final'):
                continue
            elif key.startswith('drug'):
                sd_old['drug.' + key] = sd_old.pop(key)
            else:
                sd_old['vnn.' + key] = sd_old.pop(key)
        torch.save(sd_old, new_path)
    else:
        print("please transfer the checkpoint into state_dict first.")


def kriss2text_ver2(src_path, vnn_info, dst_path='./ckpt/dc_text_v2.pt', embed_size=768):
    with open(src_path, 'rb') as f:
        datas = pickle.load(f)
        ckpt = OrderedDict()
        terms = get_terms(vnn_info)
        tmp_ckpt = dict()
        term_num = dict()
        for term in terms:
            key = term
            tmp_ckpt[key] = torch.zeros([embed_size])
            term_num[key] = 0

        for meta, data in tqdm(datas):
            ids = meta['cuis']
            embeddings = torch.from_numpy(data)
            for go_id in ids:
                tmp_ckpt[go_id] += embeddings
                term_num += 1

        res = 0
        for key in term_num.keys():
            if term_num[key] > 0:
                ckpt['vnn.%s_text-feature' %
                     key] = tmp_ckpt[key] / term_num[key]
                res += 1
            else:
                #print('Term %s has no feature embedding' % key)
                continue
        torch.save(ckpt, dst_path)
        print('There are %d in %d terms has embeddings' %
              (res, len(terms_num)))


def get_terms(vnn_info):
    terms = [line.strip() for line in open(vnn_info, 'r', encoding='utf-8')]
    return terms


if __name__ == '__main__':
    path = './ckpt/dc_v0.pt'
    src_path = './data/raw/embeddings'
    vnn_info = 'data/go_text/vnn.txt'
    # transfer(path)
    kriss2text_ver2(src_path, vnn_info)

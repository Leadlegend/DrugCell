import torch
import pickle
import random

from tqdm import tqdm
from copy import deepcopy
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


def kriss2text_ver2(src_path,
                    vnn_info,
                    dst_path='./ckpt/dc_text_v2.pt',
                    embed_size=768):
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
                if go_id in terms:
                    tmp_ckpt[go_id] += embeddings
                    term_num[go_id] += 1

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
        print('There are %d in %d terms has embeddings' % (res, len(term_num)))
        return ckpt


def get_terms(vnn_info):
    terms = [line.strip() for line in open(vnn_info, 'r', encoding='utf-8')]
    return terms


def merge_ckpt(text_ver1, text_ver2, dst_path='./ckpt/text/dc_text_v2.1.pt'):
    ckpt1 = torch.load(text_ver1)
    ckpt2 = torch.load(text_ver2)
    s = 0
    for k, v in ckpt1.items():
        if k not in ckpt2:
            ckpt2[k] = v
            s += 1
    print('Merging %d params into the new checkpoint' % s)
    torch.save(ckpt2, dst_path)


def randomize_ckpt(ckpt_old, ckpt_new, vnn_term_path):
    terms = [line.strip() for line in open(vnn_term_path)]
    term_keys = deepcopy(terms)
    random.shuffle(terms)
    randomize_term = dict()
    for i, term in enumerate(term_keys):
        randomize_term['vnn.%s_text-feature' %
                       term] = 'vnn.%s_text-feature' % terms[i]

    ckpt = torch.load(ckpt_old)
    new_ckpt = OrderedDict()
    for key, value in tqdm(randomize_term.items()):
        new_ckpt[key] = ckpt[value]

    torch.save(new_ckpt, ckpt_new)


if __name__ == '__main__':
    path = './ckpt/dc_v0.pt'
    src_path = 'data/raw/embeddings'
    vnn_info = 'data/go_text/vnn.txt'
    text_ver1 = 'ckpt/text/dc_text_v1.pt'
    text_ver2 = 'ckpt/text/dc_text_v2.pt'
    text_rand = 'ckpt/text/dc_text_rand.pt'
    vnn_small = 'data/go_text/vnn_small.txt'
    # transfer(path)
    #kriss2text_ver2(src_path, vnn_info)
    #merge_ckpt(text_ver1, text_ver2)
    randomize_ckpt(text_ver2, text_rand, vnn_small)

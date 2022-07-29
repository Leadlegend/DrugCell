import os
import re
import json
import linecache

from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional, Union
from collections import defaultdict
from config import *


def show_vnn_terms(vnn_path, vnn_term_path):
    terms = set()
    f = open(vnn_path, 'r', encoding='utf-8')
    for line in f:
        ts = line.strip().split('\t')
        assert len(ts) == 3
        b = 2 + (ts[-1] == 'default')
        ts = ts[:b]
        ts = [t for t in ts if t.startswith('GO:')]
        terms.update(ts)
    terms = list(terms)
    print("Overall Number is %d" % len(terms))
    res = '\n'.join(terms)
    g = open(vnn_term_path, 'w', encoding='utf-8')
    g.write(res)
    f.close()
    g.close()


def show_vnn_info(t2n_path, go_path, vnn_term_path, vnn_stats_path, vnn_nonzero_path=None):
    f = open(t2n_path, 'r', encoding='utf-8')
    g = open(go_path, 'r', encoding='utf-8')
    vnn_ids = [
        line.strip()
        for line in open(vnn_term_path, 'r', encoding='utf-8')
    ]
    term2num = json.load(f)
    id2term = dict()
    for line in g.readlines():
        data = json.loads(line.strip())
        id2term[data['id']] = data['name']
        alt_ids = data.get('alt_id', [])
        for alt_id in alt_ids:
            id2term[alt_id] = data['name']
    f.close()
    g.close()

    non_zero = 0
    num_sum = 0
    vnn_infos = list()

    if vnn_nonzero_path is not None:
        k = open(vnn_nonzero_path, 'w', encoding='utf-8')

    for vnn_id in vnn_ids:
        if vnn_id not in id2term:
            print('Term %s is obsolete' % vnn_id)
            continue
        term_name = id2term[vnn_id]
        if term_name not in term2num:
            print('Bad Term %s: %s' % (vnn_id, term_name))
            term_num = 0
        else:
            term_num = term2num[term_name]
        vnn_infos.append((term_name, term_num))
        if term_num > 0:
            non_zero = non_zero + 1
            num_sum += term_num
            if vnn_nonzero_path is not None:
                k.write(vnn_id + '\n')

    vnn_infos.sort(key=lambda x: x[1], reverse=True)
    vnn_infos = ["%s\t%s" % (x[0], x[1]) for x in vnn_infos]
    print('There are %d non-zero terms in %d VNN terms' %
          (non_zero, len(vnn_infos)))
    print("The average number of mention is %f" % (num_sum / len(vnn_infos)))
    if vnn_stats_path is not None:
        res = '\n'.join(vnn_infos)
        h = open(vnn_stats_path, 'w', encoding='utf-8')
        h.write(res)
        h.close()


def record_vnn_termname(vnn_term_path, go_path, output_path):
    f = open(output_path, 'w', encoding='utf-8')
    g = open(go_path, 'r', encoding='utf-8')
    vnn_ids = [
        line.strip()
        for line in open(vnn_term_path, 'r', encoding='utf-8')
    ]
    id2term = dict()
    for line in g.readlines():
        data = json.loads(line.strip())
        id2term[data['id']] = data['name']
        alt_ids = data.get('alt_id', [])
        for alt_id in alt_ids:
            id2term[alt_id] = data['name']
    g.close()
    
    vnn_info = defaultdict(list)
    for vnn_id in vnn_ids:
        vnn_info[id2term[vnn_id]].append(vnn_id)
    
    f.write(json.dumps(vnn_info, indent=4, sort_keys=True))

def main():
    #show_vnn_terms(go_path, vnn_path)
    show_vnn_info(t2n_path, go_path, vnn_term_path, vnn_stats_path, vnn_nonzero_path)
    #record_vnn_termname(vnn_term_path, go_path, output_path=vnn_info_path)


if __name__ == '__main__':
    main()

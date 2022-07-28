import re
import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from collections import defaultdict, OrderedDict


class GeneOntologyReader(object):
    def __init__(self, path):
        self.path = path
        self.total = list()
        self.read_storage_data()

    def __len__(self):
        return len(self.total)

    def read_storage_data(self):
        gt = defaultdict(list)
        f = open(self.path, 'r', encoding="utf-8")
        for line in tqdm(f.readlines()):
            if line[0] not in ['[', '\n']:
                label = line.split(":")[0]
                if label not in ['def', 'synonym']:
                    value = line[len(label)+1:].strip()
                    value = value.replace('"', "'")
                else:
                    value = line[len(label)+1:].strip().split('" ')[0][1:]
                    value = value.replace('"', "'")
                assert not value.startswith('"')
                if label not in ['is_a', 'synonym', 'alt_id', 'consider', 'xref', 'subset', 'intersection_of', 'relationship']:
                    gt[label] = value
                else:
                    gt[label].append(value)

            elif line[0] == "[" and len(gt):
                self.total.append(gt)
                gt = defaultdict(list)


def transfer_obo2json(src_path, dst_path):
    reader = GeneOntologyReader(path=src_path)
    reader.read_storage_data()
    with open(dst_path, 'w') as fileObject:
        for i in reader.total:
            line = json.dumps(i) + '\n'
            fileObject.write(line)
        fileObject.close()


def term2text(term: dict):
    text = ''
    text += term['name'] + '. '
    if 'synonym' in term:
        for syn in term['synonym']:
            text += syn + '. '
    text += term['def']
    return text


def text_analyzer(go_path, graph_path, def_path, nodef_path):
    """
    Analyze the text of a given ontology file.
    :param go_path: path to the GO info file
    :param graph_path: path to the Dcell graph file
    :param def_path: path to the output file
    """
    with open(graph_path, 'r', encoding='utf-8') as f:
        go_terms = set()
        for line in tqdm(f.readlines()):
            terms = line.strip().split('\t')[:-1]
            assert len(terms) == 2
            if not terms[1].startswith('GO:'):
                terms = terms[:1]
            go_terms.update(terms)
        f.close()
    print('Number of terms in Dcell graph: %s' % len(go_terms))
    go_def = OrderedDict()
    for term in go_terms:
        go_def[term] = False
    with open(go_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            term = json.loads(line.strip())
            if 'def' not in term:
                continue
            if term['id'] in go_def.keys():
                go_def[term['id']] = term2text(term)
            alt_ids = term.get('alt_id', [])
            for alt_id in alt_ids:
                if alt_id in go_def and go_def[alt_id] is False:
                    go_def[alt_id] = term2text(term)
        f.close()
    with open(def_path, 'w', encoding='utf-8') as f:
        with open(nodef_path, 'w', encoding='utf-8') as g:
            for term_id, term_def in go_def.items():
                if term_def is not False:
                    line = '\t'.join([term_id, term_def]) + '\n'
                    f.write(line)
                else:
                    line = term_id + '\n'
                    g.write(line)
            g.close()
        f.close()


def check_article(line):
    return len(line) > 10 and line[9] == 'a'


def test(src_path, dst_path):
    f = open(src_path, 'r', encoding='utf-8')
    g = open(dst_path, 'w', encoding='utf-8')
    for idx in tqdm(range(356922799)):
        line = f.readline()
        if check_article(line):
            g.write(line[11:])


def iter_count(file_name):
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer)
                            for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)


def art2sent(src_path, dst_path, file_len=20116988):
    f = open(src_path, 'r', encoding='utf-8')
    g = open(dst_path, 'w', encoding='utf-8')
    for i in tqdm(range(file_len)):
        art = f.readline().strip()
        if not len(art):
            continue
        sentences = sent_tokenize(art)
        sentences = [clear_sent(s) for s in sentences]
        if i <= 5:
            for sent in sentences:
                print(sent)
        res = '\n'.join(sentences)
        res += '\n'
        g.write(res)


p = re.compile(r'[(](.*?)[)]', re.S)


def clear_sent(s):
    s = re.sub(p, '', s)
    s = s.replace('\t', ' ').replace(
        '    ', ' ').replace('   ', ' ').replace('  ', ' ')
    return s


def clear_article():
    src = '/Users/iris/Documents/pku/research/Drugcell/data/raw/article.txt'
    dst = '/Users/iris/Documents/pku/research/Drugcell/data/raw/sent2.txt'
    #line_num = iter_count(src)
    # print(line_num)
    #art2sent(src, dst, line_num)
    art2sent(src, dst)


def term2keyword(src_path, dst_path):
    f = open(src_path, 'r', encoding='utf-8')
    g = open(dst_path, 'w', encoding='utf-8')
    for line in tqdm(f.readlines()):
        term = json.loads(line.strip())
        if term['name'].startswith('obsolete '):
            continue
        keyword = set()
        keyword.add(term['name'])
        keyword.update(term.get('synonym', []))
        res = {'id': term['id'], 'name': term['name'],
               'keyword': list(keyword)}
        res = json.dumps(res) + '\n'
        g.write(res)


if __name__ == '__main__':
    """
    print(iter_count(
        'data/raw/sent.txt'))
        """
    term2keyword(src_path='data/go_text/gene_ontology.txt',
                 dst_path='data/go_text/keyword.txt')
    #test('/Users/iris/Documents/pku/research/Drugcell/data/raw/bioconcepts2pubtator_offsets.txt', 'article.txt')

    # clear_article()
    #transfer_obo2json(src_path='data/raw/gotext_raw.txt', dst_path='data/go_text/gene_ontology.txt')
    #text_analyzer(go_path='data/go_text/gene_ontology.txt', graph_path='data/drugcell_ont.txt', def_path='data/go_text/definition.txt', nodef_path='data/go_text/nodef.txt')

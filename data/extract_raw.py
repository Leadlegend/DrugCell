import json
from tqdm import tqdm
from collections import defaultdict


class GeneOntologyReader(object):

    def __init__(self, path):
        self.path = path
        self.total = []

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
    go_def = {term: False for term in go_terms}
    with open(go_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            term = json.loads(line.strip())
            if 'def' not in term:
                continue
            if term['id'] in go_def.keys():
                go_def[term['id']] = term['def']
            alt_ids = term.get('alt_id', [])
            for alt_id in alt_ids:
                if alt_id in go_def and go_def[alt_id] is False:
                    go_def[alt_id] = term['def']
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


if __name__ == '__main__':
    #transfer_obo2json(src_path='data/raw/gotext_raw.txt', dst_path='data/go_text/gene_ontology.txt')
    text_analyzer(go_path='data/go_text/gene_ontology.txt', graph_path='data/drugcell_ont.txt', def_path='data/go_text/definition.txt', nodef_path='data/go_text/nodef.txt')

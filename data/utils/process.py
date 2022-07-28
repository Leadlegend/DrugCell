import json
import collections

from tqdm import tqdm


stopwords = [x.strip() for x in open('data/go_text/st.txt', 'r').readlines()]


def classify(path, keyword_path, output_path1, output_path2, output_path3, st_path=None):
    g = open(keyword_path, 'r', encoding='utf-8')
    f = open(path, 'r', encoding='utf-8')
    data = json.load(f)
    res = collections.defaultdict(set)
    for line in tqdm(g.readlines()):
        term = json.loads(line.strip())
        keywords = [k.lower()
                    for k in term['keyword'] if k.lower() not in stopwords]
        keys = list(data.keys())
        for key in keys:
            if key in keywords:
                res[term['name']].update(data[key])
                data.pop(key)
    f.close()
    g.close()

    if st_path is not None:
        print("processing stop words in %s" % st_path)
        g = open(keyword_path, 'r', encoding='utf-8')
        k = open(st_path, 'r', encoding='utf-8')
        data = json.load(k)
        for line in tqdm(g.readlines()):
            term = json.loads(line.strip())
            keywords = [k.lower()
                        for k in term['keyword'] if k.lower() in stopwords]
            keys = list(data.keys())
            for key in keys:
                if key in keywords:
                    res[term['name']].update(data[key])
        k.close()
        g.close()

    term_line = {key: list(val) for key, val in res.items()}
    term_num = {key: len(val) for key, val in res.items()}
    term_num_pos = {k: v for k, v in term_num.items() if v > 0}
    print('non-zero term count: %d in %d' %
          (len(term_num_pos), len(term_num)))

    res_line = json.dumps(term_line, indent=2, sort_keys=True)
    res_num = json.dumps(term_num, indent=2, sort_keys=True)
    term_num = [(k, v) for k, v in term_num.items()]
    term_num.sort(key=lambda x: x[1], reverse=True)
    res_num_txt = ["%s\t%s" % (x[0], x[1]) for x in term_num]
    res_num_txt = '\n'.join(res_num_txt)
    f = open(output_path1, 'w', encoding='utf-8')
    g = open(output_path2, 'w', encoding='utf-8')
    h = open(output_path3, 'w', encoding='utf-8')
    f.write(res_line)
    g.write(res_num)
    h.write(res_num_txt)
    f.close()
    g.close()
    h.close()


def raw2txt(path, dst_path):
    f = open(path, 'r', encoding='utf-8')
    g = open(dst_path, 'w', encoding='utf-8')
    data = json.load(f)
    num = [(k, len(v)) for k, v in data.items()]
    num.sort(key=lambda x: x[1], reverse=True)
    num = ['%s\t%s' % (x[0], x[1]) for x in num]
    num = '\n'.join(num)
    g.write(num)
    f.close()
    g.close()


def out2txt(path, dst_path, go_path):
    f = open(path, 'r', encoding='utf-8')
    g = open(dst_path, 'w', encoding='utf-8')
    res = list()
    data = json.load(f)
    f.close()
    for line in h.readlines():
        term = json.loads(line.strip())
        name = term['name']
        num = data.get(term['id'], 0)
        res.append((name, num))
    h.close()

    res.sort(key=lambda x: x[1], reverse=True)
    for i in res:
        i = [str(x) for x in i]
        i = '\t'.join(i) + '\n'
        g.write(i)


def generate_stop_vocab(vocab_path, keyword_path):
    f = open(keyword_path, 'r', encoding='utf-8')
    g = open(vocab_path, 'w', encoding='utf-8')
    for line in f.readlines():
        term = json.loads(line.strip())
        for key in term['keyword']:
            if key.lower() in stopwords:
                g.write(key + '\n')

    f.close()
    g.close()


if __name__ == '__main__':
    path = './res.json'
    st_path = './st_res.json'
    dst_path = './res.txt'
    go_path = '/Users/iris/Documents/pku/research/Drugcell/data/go_text/gene_ontology.txt'

    keyword_path = '/Users/iris/Documents/pku/research/Drugcell/data/go_text/keyword.txt'
    output_path1 = '/Users/iris/Documents/pku/research/Drugcell/data/go_text/term2line.json'
    output_path2 = '/Users/iris/Documents/pku/research/Drugcell/data/go_text/term2num.json'
    output_path3 = '/Users/iris/Documents/pku/research/Drugcell/data/go_text/term2num.txt'
    vocab_path = '/Users/iris/Documents/pku/research/Drugcell/data/go_text/st_vocab.txt'
    classify(path, keyword_path, output_path1, output_path2, output_path3)
    #raw2txt(path, dst_path)
    #generate_stop_vocab(vocab_path, keyword_path)
    #out2txt(path=output_path2, dst_path=output_path3, go_path=go_path)

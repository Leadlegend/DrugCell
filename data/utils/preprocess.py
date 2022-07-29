import os
import re
import json
import linecache

from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional, Union
from collections import defaultdict
from config import *


@dataclass
class Mention:
    cuis: List[str]
    start: int
    end: int
    text: str
    types: str


@dataclass
class ContextualMention:
    mention: str
    cuis: List[str]
    ctx_l: str
    ctx_r: str

    def __str__(self):
        d = dict()
        d['context_left'] = self.ctx_l
        d['context_right'] = self.ctx_r
        d['mention'] = self.mention
        d['cuis'] = self.cuis
        return json.dumps(d) + '\n'


@dataclass
class Term:
    name: str
    id: List[str]
    keyword: List[str]

    def __str__(self):
        d = dict()
        d["name"] = self.name
        d["id"] = self.id
        d["keyword"] = self.keyword
        return json.dumps(d) + '\n'


@dataclass
class Document:
    mentions: List[Mention]
    id: str = None
    title: str = None
    abstract: str = None

    def concatenate_text(self) -> str:
        return self.abstract

    def to_contextual_mentions(self,
                               max_length: int = 32
                               ) -> List[ContextualMention]:
        text = self.concatenate_text()
        mentions = []
        for m in self.mentions:
            assert m.text == text[m.start:m.end], 'Bad Mention:\n %s\n%s' % (
                m.text, text[m.start:m.end])
            # Context
            ctx_l, ctx_r = text[:m.start].strip().split(), text[m.end:].strip(
            ).split()
            ctx_l, ctx_r = ' '.join(ctx_l[-max_length:]), ' '.join(
                ctx_r[:max_length])
            cm = ContextualMention(
                mention=m.text,
                cuis=m.cuis,
                ctx_l=ctx_l,
                ctx_r=ctx_r,
            )
            mentions.append(cm)
        return mentions


def getLine(path, line_idx: int) -> str:
    return linecache.getline(path, line_idx + 1)


def reverse_term2line(t2l_path, l2t_path, vnn_info_path=None):
    f = open(t2l_path, 'r', encoding='utf-8')
    g = open(l2t_path, 'w', encoding='utf-8')
    term2line = json.load(f)
    f.close()

    if vnn_info_path is not None:
        with open(vnn_info_path, 'r', encoding='utf-8') as f:
            vnn_info = json.load(f)
            term2line = {k: v for k, v in term2line.items() if k in vnn_info}
            print('The remaining term is %d' % len(term2line))

    line2term = defaultdict(list)
    for k, v in term2line.items():
        for line in v:
            line2term[line].append(k)
    res = json.dumps(line2term, sort_keys=True, indent=4)
    g.write(res)
    g.close()
    print("Finished Reverse Term2Line.")
    return line2term


def parse_term_info(keyword_path, vnn_info_path):
    infos = dict()
    f = open(keyword_path, 'r', encoding='utf-8')
    if vnn_info_path is not None:
        vnn_info_path = json.load(open(vnn_info_path, 'r', encoding='utf-8'))
    for line in tqdm(f.readlines()):
        data = json.loads(line.strip())
        if vnn_info_path is None or data['name'] in vnn_info_path:
            term = Term(**data)
            if vnn_info_path is None:
                term.id = [term.id]
            else:
                term.id = vnn_info_path[term.name]
            infos[term.name] = term
    f.close()
    print('Term_infos has %d Term' %len(infos))
    return infos


def dic2docs(l2t_path, snt_path, cm_path, keyword_path, vnn_info_path=None):
    f = open(l2t_path, 'r', encoding='utf-8')
    g = open(cm_path, 'w', encoding='utf-8')
    line2term = json.load(f)
    f.close()
    term_infos = parse_term_info(keyword_path, vnn_info_path)

    print("Finished Parsing Term info, start creating contextual mentions...")
    for idx, (line_idx, terms) in enumerate(tqdm(line2term.items())):
        line = getLine(snt_path, int(line_idx))
        low_line = line.lower()
        mentions = list()
        for term in terms:
            term_flag = False
            term_info = term_infos[term]
            for key in term_info.keyword:
                lkey = key.lower()
                start = low_line.find(lkey)
                if start > -1:
                    end = start + len(lkey)
                    m = Mention(cuis=term_info.id,
                                start=start,
                                end=end,
                                text=line[start:end],
                                types=(term_info.name == key))
                    mentions.append(m)
                    term_flag = True
                    if idx < 5:
                        print(str(m))
                else:
                    continue
            if not term_flag:
                print("Can't find any keyword of Term %s in line:\n %s" %
                      (term, line))

        if not len(mentions):
            print("Bad line:\n %s" % line)
        else:
            doc = Document(id=line_idx,
                           title='',
                           abstract=line,
                           mentions=mentions)
            con_mentions = doc.to_contextual_mentions()
            con_mentions = [str(m) for m in con_mentions]
            res = ''.join(con_mentions) + '\n'
            g.write(res)
    g.close()
    print("Finish Data Preprocessing!")


def main():
    #reverse_term2line(t2l_path, l2t_path, vnn_info_path)
    dic2docs(l2t_path, snt_path, cm_path, keyword_path, vnn_info_path)


if __name__ == '__main__':
    main()

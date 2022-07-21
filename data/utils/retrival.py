import os
import re
import json
import collections

from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
from nltk.tokenize import word_tokenize, SpaceTokenizer


def load_vocab(vocab_file, has_index=False):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
        if not has_index:
            for index, token in enumerate(tokens):
                token = token.rstrip("\n")
                vocab[token] = index
        else:
            for i, datas in enumerate(tokens):
                index, token = datas.rstrip('\n').split('\t')
                try:
                    index = int(index)
                except:
                    raise ValueError(
                        'Invalid Vocabulary Item %s at line %d' % (datas, i))
                vocab[token] = index
    return vocab


class Tokenizer:
    def __init__(self, vocab_file: str, has_index: bool = False, unk_token=0):
        if not os.path.exists(vocab_file):
            raise ValueError("Can't Find Vocabulary File %s" % vocab_file)
        self.tk = SpaceTokenizer()
        self.unk_token = unk_token
        self.vocab = load_vocab(vocab_file, has_index=has_index)

    def tokenize(self, text):
        return self.tk.tokenize(text)

    def convert_tokens_to_ids(self, tokens: Union[List[str], str]) -> Union[int, List[int]]:
        if tokens is None:
            return None
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        ids = list()
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token)

    def __call__(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)
        return ids


class Keyword:
    def __init__(self, term: dict):
        self.id = term['id']
        self.keys = term['keyword']
        self.sent_ids = list()

    def __len__(self) -> int:
        return len(self.keys)

    def _init_keys(self, tokenizer: Tokenizer):
        self.keys = [tuple(tokenizer(x)) for x in self.keys]

    def __str__(self):
        d = {'id': self.id, 'match_num': len(
            self.sent_ids), 'sentence_id': self.sent_ids}
        return json.dumps(d) + '\n'


@dataclass
class Sentence:
    sent_id: int
    sent: str

    def _init_sent(self, tokenizer: Tokenizer):
        self.sent = tokenizer(self.sent)
        self.flag = sum(self.sent) > 0
        self.grams = dict()
        if not self.flag:
            return self.flag
        self.max_n = min(max_N, len(self.sent))
        for i in range(1, self.max_n+1):
            self._generate_ngrams(i)
        return self.flag

    def _generate_ngrams(self, n):
        ngrams = [tuple(self.sent[i:i+n])
                  for i in range(0, len(self.sent)-n+1)]
        self.grams[n] = set(ngrams)

    def check_keyword(self, key: Keyword):
        for k in key.keys:
            n = len(k)
            if n <= self.max_n and k in self.grams[n]:
                return True
            elif n > len(self.sent):
                continue
            elif n > self.max_n:
                continue
        return False

    def __del__(self):
        self.grams.clear()
        self.sent.clear()


class Searcher:
    def __init__(self, tokenizer: Tokenizer, keywords: str):
        self.tokenizer = tokenizer
        self.keywords = keywords
        self._init_keys()

    def _init_keys(self):
        print("initializing keywords in %s" % self.keywords)
        with open(self.keywords, 'r') as f:
            keys = []
            for line in tqdm(f.readlines()):
                data = json.loads(line.strip())
                key = Keyword(data)
                key._init_keys(tokenizer=self.tokenizer)
                keys.append(key)
            self.keywords = keys

    @property
    def num_keyword(self):
        return len(self.keywords)

    def search(self, sentence: str, sent_id: int = -1):
        sentence = Sentence(sent_id, sentence)
        flag = sentence._init_sent(self.tokenizer)

        matched = 0
        if flag is False:
            return matched
        for keyword in self.keywords:
            flag = sentence.check_keyword(keyword)
            if flag:
                keyword.sent_ids.append(sentence.sent_id)
                matched += 1
        sentence.__del__()
        return matched

    def print_result(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for key in self.keywords:
                f.write(str(key))
            f.close()


max_N = 14


def search():
    vocab_path = '/Users/iris/Documents/pku/research/Drugcell/data/go_text/key_vocab.txt'
    keyword_path = '/Users/iris/Documents/pku/research/Drugcell/data/go_text/keyword.txt'
    sent_path = '/Users/iris/Documents/pku/research/Drugcell/data/raw/sent.txt'
    output_path = './res.txt'
    sent_num = 128685807

    tk = Tokenizer(vocab_path, has_index=False)
    searcher = Searcher(tk, keyword_path)
    with open(sent_path, 'r', encoding='utf-8') as f:
        for idx in tqdm(range(sent_num)):
            sent = f.readline().strip().lower()
            searcher.search(sent, idx)
            if not idx % 100000:
                searcher.print_result(path=output_path)
                print('saved checkpoints after finishing %d sentences' % idx)

        searcher.print_result(path=output_path)


if __name__ == '__main__':
    search()

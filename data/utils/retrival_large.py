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
        self.reverse_vocab = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

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

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[int, List[int]]:
        if ids is None:
            return None
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = list()
        for id in ids:
            tokens.append(self._convert_id_to_token(id))
        return tokens

    def _convert_id_to_token(self, id: int) -> str:
        return self.reverse_vocab.get(id, 'UNK')


@dataclass
class Sentence:
    sent_id: int
    sent: str

    def _init_sent(self, tokenizer: Tokenizer, gram_num: int = -1):
        self.sent = tokenizer(self.sent)
        self.flag = sum(self.sent) > 0
        self.grams = dict()
        if not self.flag:
            return self.flag
        if not gram_num > 0:
            self.max_n = min(max_N, len(self.sent))
        else:
            self.max_n = min(gram_num, len(self.sent))
        for i in range(1, self.max_n+1):
            self._generate_ngrams(i)
        return self.flag

    def _generate_ngrams(self, n):
        ngrams = [tuple(self.sent[i:i+n])
                  for i in range(0, len(self.sent)-n+1)]
        self.grams[n] = set(ngrams)

    def __del__(self):
        self.grams.clear()
        self.sent.clear()


class BetterSearcher:
    def __init__(self, tokenizer: Tokenizer, keywords: str):
        self.tokenizer = tokenizer
        self.keywords = keywords
        if isinstance(keywords, str):
            self._init_keys()

    def _init_keys(self):
        print("initializing keywords in %s" % self.keywords)
        with open(self.keywords, 'r') as f:
            keys = dict()
            for idx, line in enumerate(tqdm(f.readlines())):
                data = json.loads(line.strip())
                for k in data['keyword']:
                    k = self.tokenizer.tokenize(k.lower())
                    k = ' '.join(k)
                    keys[k] = []
            self.keywords = keys

    @property
    def num_keyword(self):
        return len(self.keywords)

    def search(self, sentence: str, sent_id: int = -1, grams: int = -1):
        sentence = Sentence(sent_id, sentence)
        flag = sentence._init_sent(self.tokenizer, gram_num=grams)

        if flag is False:
            return 0
        for n, gram_set in sentence.grams.items():
            for gram in gram_set:
                if 0 in gram:
                    continue
                gram = self.tokenizer.convert_ids_to_tokens(gram)
                gram = ' '.join(gram)
                if gram in self.keywords:
                    self.keywords[gram].append(sent_id)

    def print_result(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            res = json.dumps(self.keywords, sort_keys=True, indent=4)
            f.write(res)
            f.close()


max_N = 25


def search():
    vocab_path = '/Users/iris/Documents/pku/research/Drugcell/data/raw/key_vocab.txt'
    keyword_path = '/Users/iris/Documents/pku/research/Drugcell/data/go_text/keyword.txt'
    sent_path = '/Users/iris/Documents/pku/research/Drugcell/data/raw/sent.txt'
    output_path = './res.json'
    sent_num = 128685807

    tk = Tokenizer(vocab_path, has_index=False)
    searcher = BetterSearcher(tk, keyword_path)
    with open(sent_path, 'r', encoding='utf-8') as f:
        for idx in tqdm(range(sent_num)):
            sent = f.readline().strip().lower()
            searcher.search(sent, idx)
            if not idx % 10000000:
                searcher.print_result(path=output_path)
                print('saved checkpoints after finishing %d sentences' % idx)

        searcher.print_result(path=output_path)
        print("finished!")


def stop_word_sreach():
    vocab_path = '/Users/iris/Documents/pku/research/Drugcell/data/go_text/st_vocab.txt'
    sent_path = '/Users/iris/Documents/pku/research/Drugcell/data/raw/sent.txt'
    output_path = './st_res.json'
    sent_num = 128685807

    tk = Tokenizer(vocab_path, has_index=False)
    keyword = dict()
    for key in tk.vocab.keys():
        if key != '[unk]':
            keyword[key] = list()
    searcher = BetterSearcher(tk, keyword)
    with open(sent_path, 'r', encoding='utf-8') as f:
        for idx in tqdm(range(sent_num)):
            sent = f.readline().strip()
            searcher.search(sent, idx, 1)
            if not idx % 5000000:
                searcher.print_result(path=output_path)
                print('saved checkpoints after finishing %d sentences' % idx)

        searcher.print_result(path=output_path)
        print("finished!")


if __name__ == '__main__':
    search()
    # stop_word_sreach()

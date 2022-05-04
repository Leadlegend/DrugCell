import os
import torch
import logging
import collections

from tqdm import tqdm
from typing import Optional, List, Union
from functools import partial
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from data.collate_fn import drugcell_collate_fn


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
            index, token = tokens.rstrip('\n').split('\t')
            vocab[token] = index
    return vocab


@dataclass
class DrugCellData:
    cell_id: int
    drug_id: int
    label: Optional[float] = None


@dataclass
class DrugCellBatch:
    cell_ids: torch.Tensor
    drug_ids: torch.Tensor
    labels: Optional[torch.Tensor] = None

    def __getitem__(self, idx: int):
        if idx:
            return self.labels
        else:
            return (self.cell_ids, self.drug_ids)

    def to(self, device):
        return ((self.cell_ids.to(device), self.drug_ids.to(device)), self.labels.to(device))


class Tokenizer:
    def __init__(self, vocab_file: str, has_index: bool, unk_token='[UNK]'):
        if not os.path.exists(vocab_file):
            raise ValueError("Can't Find Vocabulary File %s" % vocab_file)
        self.unk_token = unk_token
        self.vocab = load_vocab(vocab_file, has_index=has_index)
        self.reverse_vocab = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

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
        return self.reverse_vocab.get(id, -1)


class DrugCellDataset(Dataset):
    def __init__(self, cfg, cell2idx, drug2idx, sep='\t'):
        super().__init__()
        self.sep = sep
        self.path: str = cfg.path
        self.lazy_mode: bool = cfg.lazy
        self.cell_tokenizer: Tokenizer = cell2idx
        self.drug_tokenizer: Tokenizer = drug2idx
        self.data_map = list()
        self.construct_dataset(self.path, self.lazy_mode)

    def __getitem__(self, idx: int) -> DrugCellData:
        if self.lazy_mode:
            return self._lazy_get(idx)
        else:
            return self._get(idx)

    def __len__(self):
        return len(self.data_map)

    def _get(self, idx):
        return self.data_map[idx]

    def _lazy_get(self, idx):
        handler = self.data_map[idx]
        data = handler.readline().strip().split(self.sep)
        return self._parse_data(data)

    def construct_dataset(self):
        if not os.path.exists(self.path):
            raise ValueError('Bad Dataset File: %s' %
                             self.path, stack_info=True)

        if not self.lazy_mode:
            with open(self.path, "r", encoding='utf-8') as f:
                for line in tqdm(f.readlines()):
                    data = line.strip().split(self.sep)
                    self.data_map.append(self._parse_data(data))
        else:
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in tqdm(f.readlines()):
                    offset = f.tell() - len(line)
                    handler = open(self.path, 'r', encoding='utf-8')
                    handler.seek(offset)
                    self.data_map.append(handler)
        f.close()

    def _parse_data(self, data: tuple) -> DrugCellData:
        """
            robust data parser for dataset construction
        """
        flag = True
        label = None
        if len(data) > 3 or len(data) < 2:
            flag = False
        if len(data) == 3:
            try:
                label = float(data[2])
            except:
                flag = False
        cell_id, drug_id = self.cell_tokenizer.convert_tokens_to_ids(
            data[0]), self.drug_tokenizer.convert_tokens_to_ids(data[1])
        if cell_id < 0 or drug_id < 0:
            flag = False
        if not flag:
            raise ValueError("Bad Data %s" % data)
        else:
            return DrugCellData(cell_id, drug_id, label)


class DrugCellDataModule:
    def __init__(self, cfg):
        self.logger = logging.getLogger('DataModule')
        self.cfg = cfg
        self.cell2idx = Tokenizer(cfg.cell2idx, has_index=True)
        self.drug2idx = Tokenizer(cfg.drug2idx, has_index=True)
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self, cfg):
        if os.path.exists(cfg.train.path):
            self.logger.info("Constructing Train Data...")
            self.train_dataset = DrugCellDataset(
                cfg.train, self.cell2idx, self.drug2idx)
        if os.path.exists(cfg.val.path):
            self.logger.info(" Constructing Validation Data...")
            self.val_dataset = DrugCellDataset(
                cfg.val, self.cell2idx, self.drug2idx)
        if os.path.exists(cfg.test.path):
            self.logger.info("Constructing Test Data...")
            self.test_dataset = DrugCellDataset(
                cfg.test, self.cell2idx, self.drug2idx)

    def train_dataloader(self):
        dataloader = DataLoader(dataset=self.train_dataset,
                                batch_size=self.cfg.train.batch_size,
                                collate_fn=partial(
                                    drugcell_collate_fn, labeled=True),
                                pin_memory=self.cfg.train.pin,
                                num_workers=self.cfg.train.workers,
                                shuffle=self.cfg.train.shuffle
                                )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(dataset=self.val_dataset,
                                batch_size=self.cfg.val.batch_size,
                                collate_fn=partial(
                                    drugcell_collate_fn, labeled=True),
                                pin_memory=self.cfg.val.pin,
                                num_workers=self.cfg.val.workers,
                                shuffle=self.cfg.val.shuffle
                                )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(dataset=self.test_dataset,
                                batch_size=self.cfg.test.batch_size,
                                collate_fn=partial(
                                    drugcell_collate_fn, labeled=False),
                                pin_memory=self.cfg.test.pin,
                                num_workers=self.cfg.test.workers,
                                shuffle=self.cfg.test.shuffle
                                )
        return dataloader

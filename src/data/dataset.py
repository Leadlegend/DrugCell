import os
import torch
import collections

from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass
from torch.utils.data import Dataset

from .tokenizer import Tokenizer


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


class DrugCellDataset(Dataset):
    def __init__(self, cfg, cell2idx, drug2idx, sep='\t'):
        super().__init__()
        self.sep = sep
        self.path: str = cfg.path
        self.lazy_mode: bool = cfg.lazy
        self.cell_tokenizer: Tokenizer = cell2idx
        self.drug_tokenizer: Tokenizer = drug2idx
        self.data_map = list()
        self.construct_dataset()

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

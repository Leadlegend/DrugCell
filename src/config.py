import hydra

from typing import Optional, List, Union
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore


@dataclass
class DatasetConfig:
    path: Union[List[str], str]
    batch_size: int = 1
    shuffle: bool = False
    pin: bool = False
    workers: int = 0
    lazy: bool = False


@dataclass
class DataConfig:
    cell2idx: str                       # path of cell2idx file
    drug2idx: str                       # path of drug2idx file
    train: Optional[DatasetConfig]
    val: Optional[DatasetConfig]
    test: Optional[DatasetConfig]


@dataclass
class TrainerConfig:
    lr: float                   # learning rate
    epoch: int                  # epoch number
    device: str                 # Cuda / cpu
    save_dir: str               # model checkpoint saving directory
    save_period: int = 1        # save one checkpoint every $save_period epoch
    ckpt: Optional[str] = None  # model initialization
    optimizer: str = 'adam'     # optimizer name
    scheduler: str = 'None'     # lr_scheduler name


@dataclass
class LoggerConfig:
    cfg_path: str
    save_dir: str = '.'


@dataclass
class VNNConfig:
    onto: str                   # path of Ontology file
    gene2idx: str               # path of gene2idx file
    cell_embed: str             # path of cell2mutation file
    gene_hid: int = 6           # number of Neuron corresponding to a term


@dataclass
class DrugConfig:
    drug_hid: List[int]         # NN structure of drug model
    drug_embed: str             # path of drug fingerprint file


@dataclass
class DrugCellConfig:
    vnn: VNNConfig
    drug: DrugConfig
    final_hid: int = 6


@dataclass
class Config:
    data: DataConfig
    model: DrugCellConfig
    trainer: TrainerConfig
    logger: LoggerConfig


@hydra.main(config_path='../config', config_name='base')
def main(cfg: Config):
    return cfg


def args_util():
    """
        Set the template of experiment parameters (in hydra.config_store)
    """
    cs = ConfigStore.instance()
    cs.store(group='trainer', name='base_drugcell_train', node=TrainerConfig)
    cs.store(group='model', name='base_drugcell', node=DrugCellConfig)
    cs.store(group='model/vnn', name='base_drugcell', node=VNNConfig)
    cs.store(group='model/drug', name='base_base', node=DrugConfig)
    cs.store(group='data', name='base_train', node=DataConfig)
    cs.store(group='logger', name='base_base', node=LoggerConfig)


if __name__ == '__main__':
    args_util()

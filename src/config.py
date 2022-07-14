import hydra
import torch.optim as opt

from functools import partial
from dataclasses import dataclass
from typing import Optional, List, Union
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore

from model.criterions import *


cfg2opt = {
    "adam": partial(opt.Adam, betas=(0.9, 0.99), eps=1e-05),
    "sgd": opt.SGD,
}

cfg2sch = {
    "None":
    None,
    "Plateau":
    partial(
        opt.lr_scheduler.ReduceLROnPlateau,
        factor=0.9,
        mode='min',
        patience=9,
        cooldown=2,
        min_lr=2e-5,
    ),
}

cfg2crt = {
    'default': DrugCellLoss,
    'text': DrugCell_Text_Regulation,
}

cfg2ep_crt = {
    'none': None,
    "pearson": Pearson_Correlation,
    'spearman': Spearman_Correlation,
}


def init_optimizer(cfg, model):
    opt, lr, sched, epc = cfg.optimizer.lower(
    ), cfg.lr, cfg.scheduler, cfg.epoch_criterion
    optimizer = cfg2opt[opt](params=model.parameters(), lr=lr)
    scheduler = cfg2sch.get(sched, None)
    epoch_criterion = cfg2ep_crt.get(epc, None)
    if scheduler is not None:
        scheduler = scheduler(optimizer=optimizer)
    return optimizer, scheduler, epoch_criterion


@dataclass
class DatasetConfig:
    path: Union[List[str], str]
    batch_size: int = 1
    shuffle: bool = False
    pin: bool = False
    workers: int = 0
    lazy: bool = False
    label: Optional[bool] = True


@dataclass
class DataConfig:
    cell2idx: str  # path of cell2idx file
    drug2idx: str  # path of drug2idx file
    train: Optional[DatasetConfig]
    val: Optional[DatasetConfig]
    test: Optional[DatasetConfig]


@dataclass
class TrainerConfig:
    lr: float  # learning rate
    epoch: int  # epoch number
    device: str  # Cuda / cpu
    save_dir: str  # model checkpoint saving directory
    save_period: int = 1  # save one checkpoint every $save_period epoch
    ckpt: Optional[str] = None  # model initialization
    optimizer: Optional[str] = 'adam'  # optimizer name
    scheduler: Optional[str] = 'None'  # lr_scheduler name
    epoch_criterion: Optional[str] = 'pearson'


@dataclass
class LoggerConfig:
    cfg_path: str
    save_dir: str = '.'


@dataclass
class VNNConfig:
    onto: str  # path of Ontology file
    gene2idx: str  # path of gene2idx file
    cell_embed: str  # path of cell2mutation file
    gene_hid: int = 6  # number of Neuron corresponding to a term
    text_dim: int = 768


@dataclass
class DrugConfig:
    drug_hid: List[int]  # NN structure of drug model
    drug_embed: str  # path of drug fingerprint file


@dataclass
class CriterionConfig:
    name: str  # name of the criterion function
    lambda_r: float = 0.2
    lambda_t: Optional[float] = 0.0


@dataclass
class DrugCellConfig:
    vnn: VNNConfig
    drug: DrugConfig
    criterion: CriterionConfig
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
    cs.store(group='model/criterion', name='base_drugcell', node=CriterionConfig)
    cs.store(group='model/vnn', name='base_drugcell', node=VNNConfig)
    cs.store(group='model/drug', name='base_base', node=DrugConfig)
    cs.store(group='data', name='base_train', node=DataConfig)
    cs.store(group='logger', name='base_base', node=LoggerConfig)


if __name__ == '__main__':
    args_util()

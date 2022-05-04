import torch
import hydra

from functools import partial
from logger import setup_logging
from config import args_util
from trainer.drugcell import DrugCellTrainer
from data.datamodule import DrugCellDataModule
from model.drugcell import DrugCellModel, NewDrugCellModel
from model.criterions import DrugCellLoss, Pearson_Correlation


cfg2opt = {"adam": partial(torch.optim.Adam, betas=(
    0.9, 0.99), eps=1e-05), "sgd": torch.optim.SGD}
cfg2sch = {"None": None}


def init_optimizer(cfg, model):
    opt, lr, sched = cfg.optimizer.lower(), cfg.lr, cfg.scheduler
    optimizer = cfg2opt[opt](params=model.parameters(), lr=lr)
    scheduler = cfg2sch[sched]
    if scheduler is not None:
        scheduler = scheduler(optimizer=optimizer)
    return optimizer, scheduler


def train(cfg):
    datamodule = DrugCellDataModule(cfg.data)
    train_loader, val_loader = datamodule.train_dataloader(), datamodule.val_dataloader()
    model = NewDrugCellModel(cfg.model)
    optim, sched = init_optimizer(cfg.trainer, model)
    trainer = DrugCellTrainer(model=model, config=cfg.trainer, device=cfg.trainer.device,
                              data_loader=train_loader, valid_data_loader=val_loader,
                              optimizer=optim, lr_scheduler=sched,
                              criterion=DrugCellLoss, epoch_criterion=Pearson_Correlation)
    trainer.train()


@hydra.main(config_path='../config', config_name='base')
def main(configs):
    setup_logging(save_dir=configs.logger.save_dir,
                  log_config=configs.logger.cfg_path)
    torch.set_printoptions(precision=5)
    train(cfg=configs)


if __name__ == '__main__':
    args_util()
    main()

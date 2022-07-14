import torch
import hydra

from logger import setup_logging
from model.drugcell import DrugCellModel
from trainer.drugcell import DrugCellTrainer
from data.datamodule import DrugCellDataModule
from config import args_util, init_optimizer


def train(cfg):
    datamodule = DrugCellDataModule(cfg.data)
    train_loader, val_loader = datamodule.train_dataloader(), datamodule.val_dataloader()
    
    model = DrugCellModel(cfg.model)
    optim, sched, epoch = init_optimizer(cfg.trainer, model)
    
    trainer = DrugCellTrainer(model=model, config=cfg.trainer, device=cfg.trainer.device,
                              data_loader=train_loader, valid_data_loader=val_loader,
                              optimizer=optim, lr_scheduler=sched, epoch_criterion=epoch)
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

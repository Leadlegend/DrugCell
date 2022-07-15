import torch
import hydra

from logger import setup_logging
from model.drugcell import DrugCellModel
from tester.drugcell import DrugCellTester
from data.datamodule import DrugCellDataModule
from config import args_util, cfg2ep_crt, cfg2crt


def cross_valid_test(cfg):
    datamodule = DrugCellDataModule(cfg.data)
    test_loader = datamodule.test_dataloader()
    model = DrugCellModel(cfg.model)
    tester = DrugCellTester(model=model, config=cfg.trainer, device=cfg.trainer.device,
                              data_loader=test_loader, criterion=cfg2crt[cfg.model.criterion.name], 
                              epoch_criterion=cfg2ep_crt.get(cfg.trainer.epoch_criterion, None))
    tester.test()


def predict(cfg):
    raise NotImplementedError

@hydra.main(config_path='../config', config_name='cross_valid_test')
def main(configs):
    setup_logging(save_dir=configs.logger.save_dir,
                  log_config=configs.logger.cfg_path,
                  file_name='test.log')
    torch.set_printoptions(precision=5)
    cross_valid_test(cfg=configs)


if __name__ == '__main__':
    args_util()
    main()

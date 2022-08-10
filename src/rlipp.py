import os
import hydra

from logger import setup_logging
from model.drugcell import DrugCellModel
from tester.drugcell import DrugCellTester
from data.datamodule import DrugCellDataModule
from config import rlipp_args_util, args_util


@hydra.main(config_path='../config', config_name='rlipp_pre')
def preprocess(args):
    setup_logging(save_dir=configs.logger.save_dir,
                  log_config=configs.logger.cfg_path,
                  file_name='pre.log')
    torch.set_printoptions(precision=5)
    data_module = DrugCellDataModule(args.data)
    test_loader = data_module.test_dataloader()
    model = DrugCellModel(args.model)
    tester = DrugCellTester(model=model, config=cfg.trainer, device=cfg.trainer.device,
                            data_loader=test_loader)
    tester.rlipp_preprocess()


@hydra.main(config_path='../config', config_name='rlipp')
def main(args):
    setup_logging(save_dir=configs.logger.save_dir,
                  log_config=configs.logger.cfg_path,
                  file_name='main.log')
    rlipp_calculator = RLIPPCalculator(args)
    rlipp_calculator.calc_scores()


def rlipp(mode='main'):
    if mode == 'main':
        rlipp_args_util()
        main()
    elif mode == 'pre':
        args_util()
        preprocess()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    rlipp('pre')

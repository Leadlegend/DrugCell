import hydra
import torch

from logger import setup_logging
from model.drugcell import DrugCellModel
from tester.drugcell import DrugCellTester
from data.datamodule import DrugCellDataModule
from config import args_util


@hydra.main(config_path='../config', config_name='rlipp_pre')
def preprocess(args):
    setup_logging(save_dir=args.logger.save_dir,
                  log_config=args.logger.cfg_path,
                  file_name='rlipp_pre.log')
    torch.set_printoptions(precision=5)
    data_module = DrugCellDataModule(args.data)
    test_loader = data_module.test_dataloader()
    model = DrugCellModel(args.model)
    tester = DrugCellTester(model=model, config=args.trainer, device=args.trainer.device,
                            data_loader=test_loader)
    tester.rlipp_preprocess()


if __name__ == '__main__':
    args_util()
    preprocess()

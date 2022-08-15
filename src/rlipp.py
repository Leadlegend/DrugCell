import hydra

from logger import setup_logging
from config import rlipp_args_util
from tester.rlipp import RLIPPCalculator


@hydra.main(config_path='../config', config_name='rlipp')
def main(args):
    setup_logging(save_dir=args.logger.save_dir,
                  log_config=args.logger.cfg_path,
                  file_name='rlipp.log')
    rlipp_calculator = RLIPPCalculator(args)
    rlipp_calculator.calc_scores()


if __name__ == "__main__":
    rlipp_args_util()
    main()

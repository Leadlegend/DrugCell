# DrugCell

This is an reproduction version of DrugCell Neural Network to better fit for experiment setup and further research.

## Dependency

This repo additionally depends on the following packages of python:

- hydra
  
  ```bash
  pip install hydra-core --upgrade
  ```

- tqdm

  ```bash
  pip install tqdm
  ```

## Usage

We use `hydra`, rather than `argparse` to implement better configuration setting up and recording, which provides a hierarchical configuration composable from multiple sources (including overriding from command line). We will simply show its usage:

1. To use the default configuration in `./config/`, simply run:

   ```bash
   python src/train.py
   ```

2. To check the current  hierarchical configuration for your experiment, run:

   ```bash
   python src/train.py --cfg job
   ```

3. To override some of the configuration on command line, you can use:

   ``` bash
   python src/train.py trainer.optimizer=sgd ~data.train.0
   ```

   where `trainer.optimizer=sgd` means to modify your optimizer as `SGD`, and `~data.train.0` means to delete the first train dataset file if you introduce a list of files as training data.

   **Notice**: Hydra ver1.1 seems to have a bug about list index overriding, you may need to modify the source code of hydra to fix it.

4. To set up your custom configuration, you can create a `YAML` file in a subfolder under `config`, say `kay/test.yaml`. Then you can modify the configuration whatever you want (while they should always follow the format described in `./src/config.py:Config`) . Finally, to build your experiments, input:

   ```bash
   python src/train.py +kay=test
   ```

5. Also, in custom override file described above (say `kay/test`), you can also employ default list to specifically modify your config. You can take `kay/cross.yaml` as an example.

6. We also implemented some bash scripts to conduct experiments with more sophisticated configuration. For example, you can conduct 5-fold cross-validation by running 

   `bash ./scripts/cross-validation.py`

## Log

Unlike the crude and simple experimental log, we implemented rather fine-grained logging for debugging and experiment recording (support for Tensorboard will be added in future versions). By default, both of the cufiguration log and experimental log will be stored in `./outputs/`.

Notably, configuration file for logger is stored by itself in `./src/logger/config.json`. You can modify its handler or formatter in it.

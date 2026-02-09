#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import os
import sys

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from noether.training.cli import setup_hydra
from noether.training.runners import HydraRunner

setup_hydra()


@hydra.main(
    config_path=None,
    config_name=None,
    version_base="1.3",
)
def main(config: DictConfig):
    """Main entry point for training.

    This script is wrapped in a hydra function to allow for easy configuration.
    It supports passing the configuration file as a positional argument or via the --hp flag.

    Example:
    .. code-block:: bash
        python main_train.py configs/my_experiment.yaml
        python main_train.py --hp configs/my_experiment.yaml
        python main_train.py configs/my_experiment.yaml trainer.max_epochs=10
    """
    # disable hydra changing working directory
    os.chdir(hydra.utils.get_original_cwd())

    # add working directory to PYTHONPATH
    sys.path.insert(0, hydra.utils.get_original_cwd())  # store this

    config = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))

    HydraRunner().run(config)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()

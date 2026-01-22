#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from noether.training.runners import HydraRunner


def get_config_dir():
    # Resolves symlinks and ensures we get the real physical path:
    path = Path(__file__).parent / "dummy_project" / "configs"
    return str(path.resolve())


def test_train_pipeline_does_not_error():
    # Clear any leftover hydra state from previous tests:
    GlobalHydra.instance().clear()

    abs_config_dir = get_config_dir()

    with initialize_config_dir(version_base=None, config_dir=abs_config_dir, job_name="test"):
        config = compose(config_name="base_experiment")

    config = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))
    HydraRunner().run(config)


def test_train_pipeline_copies_code(tmp_path: Path):
    GlobalHydra.instance().clear()

    abs_config_dir = get_config_dir()

    with initialize_config_dir(version_base=None, config_dir=abs_config_dir, job_name="test"):
        config = compose(config_name="base_experiment")

    config = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))
    config["store_code_in_output"] = True
    config["output_path"] = tmp_path.as_posix()

    HydraRunner().run(config)

    output_dirs = list(tmp_path.iterdir())
    assert len(output_dirs) == 1 and output_dirs[0].is_dir()
    assert (output_dirs[0] / "code.tar.gz").exists()

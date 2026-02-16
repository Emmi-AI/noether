#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import os
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _rewrite_hydra_args(relpath, insert_at: int):
    hp = Path(os.getcwd()) / relpath
    if not hp.exists():
        raise FileNotFoundError(f"--hp file does not exist ('{hp.as_posix()}')")

    cn = hp.name
    cp = hp.parent.as_posix()
    sys.argv.insert(insert_at, "-cp")
    sys.argv.insert(insert_at + 1, cp)
    sys.argv.insert(insert_at + 2, "-cn")
    sys.argv.insert(insert_at + 3, cn)

    # backup the --hp argument to log it to the tracker (hydra operates on absolute path)


def setup_hydra():
    # allow --hp examples/yamls/train.yaml instead of having to specify -cp /home/USER/core/examples/yamls --cn train.yaml
    if len(sys.argv) < 2:
        print("Provide path to a config yaml file as first argument or --hp <path>", file=sys.stderr)
        sys.exit(1)

    # Check if we should add custom help
    if "--help" in sys.argv or "-h" in sys.argv:
        prog_name = Path(sys.argv[0]).name
        runner_name = "Training" if "train" in prog_name else "Inference" if "inference" in prog_name else "Generic"

        help_header = (
            f"Noether {runner_name} Runner\n"
            "----------------------\n"
            f"This runner allows you to run {runner_name.lower()} experiments using YAML configurations.\n\n"
            "Usage:\n"
            f"  python {prog_name} <config_path.yaml> [overrides]\n"
            f"  python {prog_name} --hp <config_path.yaml> [overrides]\n\n"
            "Arguments:\n"
            "  config_path.yaml  Path to the experiment configuration.\n"
            "  --hp              Flag to specify the experiment configuration path.\n"
            "  overrides         Standard Hydra overrides (e.g. trainer.max_epochs=10).\n"
        )
        sys.argv.append(f"hydra.help.header='{help_header}'")

    if sys.argv[1].endswith(".yaml"):
        relpath = sys.argv.pop(1)
        _rewrite_hydra_args(relpath, 1)
    else:
        i = 0
        while i < len(sys.argv) - 1:
            if sys.argv[i] == "--hp":
                if not sys.argv[i + 1].endswith(".yaml"):
                    raise ValueError(f"invalid --hp file {sys.argv[i + 1]}")
                sys.argv.pop(i)  # remove --hp
                relpath = sys.argv.pop(i)
                _rewrite_hydra_args(relpath, i)
                break
            i += 1

    # Disable hydra creating output dir and storing config in output folder.
    sys.argv.extend(
        ["hydra.run.dir=.", "hydra.output_subdir=null", "hydra/hydra_logging=disabled", "hydra/job_logging=disabled"]
    )

    # allow collator: ${seed:${vars.collator},0} in yaml
    def seed(x: DictConfig, y: int) -> DictConfig:
        assert isinstance(x, DictConfig)
        assert "seed" not in x
        assert isinstance(y, int)
        return DictConfig(content=dict(**x, seed=y))

    OmegaConf.register_new_resolver("seed", seed)

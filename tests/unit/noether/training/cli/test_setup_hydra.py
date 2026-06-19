#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import sys

import pytest
from hydra._internal.utils import get_args_parser
from omegaconf import OmegaConf

from noether.training.cli import setup_hydra


@pytest.fixture
def hp_file(tmp_path, monkeypatch):
    """A minimal config yaml; cwd is moved to tmp_path so relative --hp paths resolve."""
    hp = tmp_path / "config.yaml"
    hp.write_text("tracker: disabled\n")
    monkeypatch.chdir(tmp_path)
    yield hp
    # setup_hydra registers the `seed` resolver; clear it so repeated calls across tests don't collide.
    OmegaConf.clear_resolver("seed")


def _assert_hydra_parseable() -> None:
    """Hydra's argparse parser must accept the rewritten argv.

    Argparse can only consume positionals (Hydra overrides) as a single contiguous
    group, so `-cp/-cn` must not be inserted between overrides.
    """
    args = get_args_parser().parse_args(sys.argv[1:])
    assert args.config_name == "config.yaml"


class TestSetupHydra:
    def test_hp_first(self, hp_file, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["noether-train", "--hp", "config.yaml", "trainer.max_epochs=1"])

        setup_hydra()

        _assert_hydra_parseable()

    def test_positional_yaml(self, hp_file, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["noether-train", "config.yaml", "trainer.max_epochs=1"])

        setup_hydra()

        _assert_hydra_parseable()

    def test_override_before_hp(self, hp_file, monkeypatch):
        """Regression: an override before --hp (e.g. `noether-eval run_dir=... --hp extra.yaml`)
        used to split Hydra's positional overrides around -cp/-cn, making argparse fail with
        `unrecognized arguments: hydra.run.dir=. ...`."""
        monkeypatch.setattr(sys, "argv", ["noether-eval", "tracker=disabled", "--hp", "config.yaml"])

        setup_hydra()

        _assert_hydra_parseable()
        # All overrides (user + appended hydra.* ones) form one contiguous positional group.
        args = get_args_parser().parse_args(sys.argv[1:])
        assert "tracker=disabled" in args.overrides
        assert "hydra.run.dir=." in args.overrides

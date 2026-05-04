#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.training_pipelines.fixtures.helpers import run_hydra_recipe

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HEAT_TRANSFER_CONFIGS = str(_REPO_ROOT / "recipes" / "heat_transfer" / "configs")
_STUB_HEATSINK = "tests.integration.training_pipelines.fixtures.synthetic_datasets.StubSimshiftHeatsinkDataset"


@pytest.mark.usefixtures("heat_transfer_on_path")
@pytest.mark.parametrize("experiment_name", ["ab_upt", "transolver"])
def test_simshift_heatsink_pipeline(experiment_name: str, tmp_path: Path, accelerator: str, device: str) -> None:
    """SIMSHIFT-Heatsink recipe runs end-to-end and updates weights."""
    run_hydra_recipe(
        configs_dir=_HEAT_TRANSFER_CONFIGS,
        config_name="train_simshift_heatsink",
        overrides=[f"+experiment/simshift_heatsink={experiment_name}"],
        stub_kind=_STUB_HEATSINK,
        tmp_path=tmp_path,
        accelerator=accelerator,
        device=device,
        label=f"simshift_heatsink/{experiment_name}",
    )

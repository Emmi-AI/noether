#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import json

import h5py
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from noether.data.datasets.cfd.simshift_heatsink.config import SimshiftHeatsinkConfig
from noether.data.datasets.cfd.simshift_heatsink.dataset import SimshiftHeatsinkDataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_NODES = 50


@pytest.fixture()
def heatsink_root(tmp_path):
    """Create a minimal SIMSHIFT heatsink dataset on disk for testing."""
    sample_ids = list(range(1, 11))  # 10 samples

    # -- splits.json --
    splits = {
        "medium": {
            "src": {
                "train": sample_ids[:5],
                "val": [sample_ids[5]],
                "test": sample_ids[6:8],
            },
            "tgt": {
                "train": sample_ids[8:9],
                "test": sample_ids[9:10],
            },
        },
        "easy": {
            "src": {
                "train": sample_ids[:5],
                "val": [sample_ids[5]],
                "test": sample_ids[6:8],
            },
            "tgt": {
                "train": sample_ids[8:9],
                "test": sample_ids[9:10],
            },
        },
        "hard": {
            "src": {
                "train": sample_ids[:5],
                "val": [sample_ids[5]],
                "test": sample_ids[6:8],
            },
            "tgt": {
                "train": sample_ids[8:9],
                "test": sample_ids[9:10],
            },
        },
    }
    with open(tmp_path / "splits.json", "w") as f:
        json.dump(splits, f)

    # -- metadata.csv --
    rows = []
    for sid in sample_ids:
        rows.append(
            {
                "sample_id": sid,
                "fins": sid + 4,
                "spacing": 0.1 * sid,
                # dropped columns (boundary conditions)
                "envTemp": 300.0,
                "flowVelocity": 1.0,
                "height1": 0.05,
                "length": 0.1,
                "pressure": 0.0,
                "turbulentKE": 0.01,
                "turbulentOmega": 1.0,
                "width": 0.05,
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "metadata.csv", index=False)

    # -- HDF5 sample files --
    rng = np.random.default_rng(42)
    for sid in sample_ids:
        with h5py.File(tmp_path / f"{sid}.h5", "w") as h5f:
            mesh_grp = h5f.create_group("mesh")
            mesh_grp.create_dataset("element_coords", data=rng.standard_normal((NUM_NODES, 3)).astype(np.float32))
            mesh_grp.create_dataset("element_connectivity", data=rng.integers(0, NUM_NODES, size=(NUM_NODES * 2, 4)))

            fields_grp = h5f.create_group("element_fields")
            fields_grp.create_dataset("U", data=rng.standard_normal((NUM_NODES, 3)).astype(np.float32))
            fields_grp.create_dataset("T", data=rng.standard_normal(NUM_NODES).astype(np.float32))
            fields_grp.create_dataset("p_rgh", data=rng.standard_normal(NUM_NODES).astype(np.float32))

    return tmp_path


def _make_config(root, **overrides):
    defaults = {
        "kind": "simshift_heatsink",
        "root": str(root),
        "split": "train",
        "difficulty": "medium",
        "domain": "source",
    }
    defaults.update(overrides)
    return SimshiftHeatsinkConfig(**defaults)


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


def test_config_valid_minimal():
    config = SimshiftHeatsinkConfig(
        kind="simshift_heatsink",
        root="/tmp/data",
        split="train",
    )
    assert config.difficulty is None
    assert config.domain is None


def test_config_invalid_difficulty():
    with pytest.raises(ValidationError):
        SimshiftHeatsinkConfig(
            kind="simshift_heatsink",
            root="/tmp/data",
            split="train",
            difficulty="extreme",
        )


def test_config_invalid_domain():
    with pytest.raises(ValidationError):
        SimshiftHeatsinkConfig(
            kind="simshift_heatsink",
            root="/tmp/data",
            split="train",
            domain="unknown",
        )


# ---------------------------------------------------------------------------
# Dataset initialization and loading tests
# ---------------------------------------------------------------------------


def test_dataset_init_source_train(heatsink_root):
    config = _make_config(heatsink_root)
    dataset = SimshiftHeatsinkDataset(dataset_config=config)
    assert len(dataset) == 5


def test_dataset_init_source_val(heatsink_root):
    config = _make_config(heatsink_root, split="val")
    dataset = SimshiftHeatsinkDataset(dataset_config=config)
    assert len(dataset) == 1


def test_dataset_init_target_train(heatsink_root):
    config = _make_config(heatsink_root, domain="target")
    dataset = SimshiftHeatsinkDataset(dataset_config=config)
    assert len(dataset) == 1


def test_dataset_init_target_val_falls_back_to_test(heatsink_root):
    """Target domain has no val split; it should fall back to test."""
    config = _make_config(heatsink_root, domain="target", split="val")
    dataset = SimshiftHeatsinkDataset(dataset_config=config)
    assert len(dataset) == 1  # same as target test


def test_dataset_getitem_keys(heatsink_root):
    config = _make_config(heatsink_root)
    dataset = SimshiftHeatsinkDataset(dataset_config=config)
    dataset.compute_statistics = True
    sample = dataset[0]
    expected_keys = {
        "index",
        "volume_position",
        "volume_velocity",
        "volume_temperature",
        "volume_pressure",
        "simulation_parameters",
    }
    assert set(sample.keys()) == expected_keys


def test_dataset_getitem_shapes(heatsink_root):
    config = _make_config(heatsink_root)
    dataset = SimshiftHeatsinkDataset(dataset_config=config)
    dataset.compute_statistics = True
    sample = dataset[0]

    assert sample["volume_position"].shape == (NUM_NODES, 3)
    assert sample["volume_velocity"].shape == (NUM_NODES, 3)
    assert sample["volume_temperature"].shape == (NUM_NODES, 1)
    assert sample["volume_pressure"].shape == (NUM_NODES, 1)
    assert sample["simulation_parameters"].shape == (2,)  # fins, spacing


def test_dataset_sample_info(heatsink_root):
    config = _make_config(heatsink_root)
    dataset = SimshiftHeatsinkDataset(dataset_config=config)
    info = dataset.sample_info(0)
    assert "sample_id" in info
    assert info["difficulty"] == "medium"
    assert info["domain"] == "source"
    assert info["split"] == "train"


def test_dataset_all_difficulties(heatsink_root):
    for difficulty in ("easy", "medium", "hard"):
        config = _make_config(heatsink_root, difficulty=difficulty)
        dataset = SimshiftHeatsinkDataset(dataset_config=config)
        assert len(dataset) == 5

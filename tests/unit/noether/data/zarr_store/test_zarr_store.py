#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests for the chunked/sharded Zarr CFD store (writer, reader, manifest, layout)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from zarr.storage import LocalStore

from noether.data.schemas import FileMap
from noether.data.zarr_store import (
    StoreManifest,
    ZarrChunkReader,
    ZarrStoreWriter,
    build_domain_layouts,
    filename_to_canonical,
)

FILEMAP = FileMap(
    surface_position="surface_points.pt",
    surface_pressure="surface_pressure.pt",
    surface_normals="surface_normals.pt",
    volume_position="volume_points.pt",
    volume_velocity="volume_velocity.pt",
    volume_distance_to_surface="volume_sdf.pt",
    volume_normals="volume_normals.pt",
)

NS, NV = 3586, 28504


def _make_sample(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "surface_position": rng.standard_normal((NS, 3)).astype("float32"),
        "surface_pressure": rng.standard_normal((NS,)).astype("float32"),
        "surface_normals": rng.standard_normal((NS, 3)).astype("float32"),
        "volume_position": rng.standard_normal((NV, 3)).astype("float32"),
        "volume_velocity": rng.standard_normal((NV, 3)).astype("float32"),
        "volume_sdf": rng.standard_normal((NV,)).astype("float32"),
        "volume_normals": rng.standard_normal((NV, 3)).astype("float32"),
    }


@pytest.fixture
def store(tmp_path: Path) -> Path:
    writer = ZarrStoreWriter(tmp_path, FILEMAP, "test", shuffle_seed=0, chunk_points=4096)
    writer.write_sample("param0/sample", _make_sample())
    writer.save_manifest()
    return tmp_path


def test_full_read_is_lossless_for_positions_and_bounded_for_fields(store: Path) -> None:
    reader = ZarrChunkReader(store)
    out = reader.read_sample("param0/sample", num_points=None)
    original = _make_sample()

    # Positions are stored float32 -> exact (compare sorted, order is shuffled).
    got = np.sort(out["volume_position"].numpy(), axis=0)
    exp = np.sort(original["volume_position"], axis=0)
    np.testing.assert_array_equal(got, exp)

    # float16 fields -> small bounded error.
    got_v = np.sort(out["volume_velocity"].numpy(), axis=0)
    exp_v = np.sort(original["volume_velocity"], axis=0)
    assert np.abs(got_v - exp_v).max() < 1e-2

    # Scalar fields come back 1-D.
    assert out["surface_pressure"].ndim == 1
    assert out["volume_sdf"].shape == (NV,)


def test_subsampling_returns_exact_point_counts(store: Path) -> None:
    reader = ZarrChunkReader(store)
    out = reader.read_sample("param0/sample", num_points={"surface": 1000, "volume": 4096})
    assert out["surface_position"].shape == (1000, 3)
    assert out["surface_pressure"].shape == (1000,)
    assert out["volume_velocity"].shape == (4096, 3)
    assert out["volume_normals"].shape == (4096, 3)


def test_coords_and_fields_stay_point_aligned(tmp_path: Path) -> None:
    """A row read from coords must correspond to the same point in the fields array."""
    n = 512
    # Encode each point's original index in column 0 of both position and velocity.
    # f16 represents integers < 2048 exactly, so the marker survives the float16 cast.
    idx = np.arange(n, dtype="float32")
    sample = {
        "surface_position": np.stack([idx, np.zeros(n), np.zeros(n)], axis=1).astype("float32"),
        "surface_pressure": idx.copy(),
        "surface_normals": np.zeros((n, 3), "float32"),
    }
    fm = FileMap(surface_position="p.pt", surface_pressure="pr.pt", surface_normals="nm.pt")
    writer = ZarrStoreWriter(tmp_path, fm, "align", chunk_points=16)
    writer.write_sample("s", sample)
    writer.save_manifest()

    reader = ZarrChunkReader(tmp_path)
    gen = torch.Generator().manual_seed(123)
    out = reader.read_sample("s", num_points={"surface": 128}, generator=gen)
    # position[:,0] is the point index; pressure carries the same index -> must match row-wise.
    np.testing.assert_array_equal(out["surface_position"][:, 0].numpy(), out["surface_pressure"].numpy())


def test_chunk_selection_is_deterministic_with_seed(store: Path) -> None:
    reader = ZarrChunkReader(store)
    a = reader.read_sample("param0/sample", num_points={"volume": 4096}, generator=torch.Generator().manual_seed(7))
    b = reader.read_sample("param0/sample", num_points={"volume": 4096}, generator=torch.Generator().manual_seed(7))
    c = reader.read_sample("param0/sample", num_points={"volume": 4096}, generator=torch.Generator().manual_seed(8))
    np.testing.assert_array_equal(a["volume_velocity"].numpy(), b["volume_velocity"].numpy())
    assert not np.array_equal(a["volume_velocity"].numpy(), c["volume_velocity"].numpy())


def test_chunk_read_transfers_far_less_than_full_read(tmp_path: Path) -> None:
    """Subsampling must read only a fraction of the full sample's bytes."""
    # Small chunks so a small draw touches few of many chunks in *both* domains
    # (surface -> 8 chunks, volume -> 56), avoiding the full-surface dilution.
    writer = ZarrStoreWriter(tmp_path, FILEMAP, "bytes", chunk_points=512)
    writer.write_sample("param0/sample", _make_sample())
    writer.save_manifest()

    class CountingLocalStore(LocalStore):
        bytes_read = 0

        async def get(self, key, prototype, byte_range=None):  # type: ignore[override]
            res = await super().get(key, prototype, byte_range)
            if res is not None and "/c/" in str(key):
                type(self).bytes_read += len(res)
            return res

    reader = ZarrChunkReader(tmp_path, store_factory=lambda path: CountingLocalStore(path, read_only=True))

    CountingLocalStore.bytes_read = 0
    reader.read_sample("param0/sample", num_points={"surface": 512, "volume": 512})
    sub_bytes = CountingLocalStore.bytes_read

    reader._groups.clear()  # drop cached handles so the full read re-fetches
    CountingLocalStore.bytes_read = 0
    reader.read_sample("param0/sample", num_points=None)
    full_bytes = CountingLocalStore.bytes_read

    # A 512-point draw touches ~1 chunk per domain out of 64 total -> well under a fifth.
    assert sub_bytes < full_bytes / 5


def test_consolidated_metadata_makes_sample_open_one_metadata_get(store: Path) -> None:
    """The writer consolidates each sample group, so a first read costs one zarr.json GET."""

    class CountingLocalStore(LocalStore):
        metadata_gets = 0

        async def get(self, key, prototype, byte_range=None):  # type: ignore[override]
            res = await super().get(key, prototype, byte_range)
            if res is not None and str(key).endswith("zarr.json"):
                type(self).metadata_gets += 1
            return res

    reader = ZarrChunkReader(store, store_factory=lambda path: CountingLocalStore(path, read_only=True))
    CountingLocalStore.metadata_gets = 0
    reader.read_sample("param0/sample", num_points={"surface": 1000, "volume": 4096})
    assert CountingLocalStore.metadata_gets == 1  # root zarr.json carries all child metadata


def test_parallel_reads_match_serial(store: Path) -> None:
    """read_concurrency > 1 must return bit-identical results to serial reads."""
    serial = ZarrChunkReader(store, read_concurrency=1)
    parallel = ZarrChunkReader(store, read_concurrency=8)
    points = {"surface": 1000, "volume": 4096}
    a = serial.read_sample("param0/sample", num_points=points, generator=torch.Generator().manual_seed(5))
    b = parallel.read_sample("param0/sample", num_points=points, generator=torch.Generator().manual_seed(5))
    assert a.keys() == b.keys()
    for field in a:
        np.testing.assert_array_equal(a[field].numpy(), b[field].numpy())


def test_read_coords_is_an_independent_position_subsample(tmp_path: Path) -> None:
    """read_coords returns a deterministic, seed-sensitive subset of a domain's positions."""
    # Small chunks so a 1000-point draw spans many chunks (genuine random selection).
    writer = ZarrStoreWriter(tmp_path, FILEMAP, "geom", chunk_points=256)
    writer.write_sample("param0/sample", _make_sample())
    writer.save_manifest()
    reader = ZarrChunkReader(tmp_path)

    g1 = reader.read_coords("param0/sample", "surface", 1000, torch.Generator().manual_seed(0))
    g2 = reader.read_coords("param0/sample", "surface", 1000, torch.Generator().manual_seed(0))
    g3 = reader.read_coords("param0/sample", "surface", 1000, torch.Generator().manual_seed(1))
    assert g1.shape == (1000, 3)
    np.testing.assert_array_equal(g1.numpy(), g2.numpy())  # deterministic for a fixed seed
    assert not np.array_equal(g1.numpy(), g3.numpy())  # a different seed draws different points

    # Every drawn position belongs to the full surface cloud.
    full = reader.read_sample("param0/sample", None)["surface_position"].numpy()
    full_rows = {tuple(row) for row in full}
    assert all(tuple(row) in full_rows for row in g1.numpy())


def test_dataset_emits_independent_geometry_position(tmp_path: Path) -> None:
    """num_geometry_points adds geometry_position as a separate surface draw."""
    from noether.data.base.dataset import StandardDatasetConfig
    from noether.data.datasets.cfd.zarr_aero_dataset import ZarrAeroDataset

    writer = ZarrStoreWriter(tmp_path, FILEMAP, "geom", chunk_points=256)
    writer.write_sample("param0/sample", _make_sample())
    writer.save_manifest()

    class _DS(ZarrAeroDataset):
        def __len__(self) -> int:
            return 1

        def _sample_id(self, idx: int) -> str:
            return "param0/sample"

    cfg = StandardDatasetConfig(root=str(tmp_path), split="train")
    ds = _DS(cfg, FILEMAP, num_points={"surface": 512, "volume": 4096}, num_geometry_points=2000, sampling_seed=0)
    assert "getitem_geometry_position" in ds.get_all_getitem_names()

    sample = ds[0]
    assert sample["geometry_position"].shape == (2000, 3)
    assert sample["surface_position"].shape == (512, 3)  # anchors: separate, smaller draw

    # Disabled by default.
    ds_off = _DS(cfg, FILEMAP, num_points={"surface": 512, "volume": 4096})
    assert "getitem_geometry_position" not in ds_off.get_all_getitem_names()
    assert "geometry_position" not in ds_off[0]


def test_zarr_config_resolves_and_parses_from_kind() -> None:
    """The dataset config is discoverable via its `kind` and parses the sampling fields."""
    from noether.core.schemas.lib import resolve_config_class
    from noether.data.base.dataset import DatasetBaseConfig
    from noether.data.datasets.cfd import ZarrShapeNetCarDatasetConfig

    kind = "noether.data.datasets.cfd.ZarrShapeNetCarDataset"
    # The framework resolves the config class from the dataset `kind` (via __init__ hints).
    assert resolve_config_class(kind, DatasetBaseConfig) is ZarrShapeNetCarDatasetConfig

    cfg = ZarrShapeNetCarDatasetConfig.model_validate(
        {"kind": kind, "root": "/store", "split": "train", "num_surface_points": 256, "num_geometry_points": 3586}
    )
    assert (cfg.root, cfg.split) == ("/store", "train")
    assert cfg.num_surface_points == 256 and cfg.num_geometry_points == 3586
    assert cfg.num_volume_points is None and cfg.read_concurrency == 1


def test_zarr_dataset_reads_sampling_from_config(tmp_path: Path) -> None:
    """Subsampling counts are driven entirely by the config object."""
    from noether.data.datasets.cfd.zarr_aero_dataset import ZarrAeroDataset, ZarrAeroDatasetConfig

    writer = ZarrStoreWriter(tmp_path, FILEMAP, "cfg", chunk_points=256)
    writer.write_sample("s", _make_sample())
    writer.save_manifest()

    class _DS(ZarrAeroDataset):
        def __init__(self, cfg: ZarrAeroDatasetConfig) -> None:
            super().__init__(
                cfg,
                FILEMAP,
                num_points={"surface": cfg.num_surface_points, "volume": cfg.num_volume_points},
                sampling_seed=cfg.sampling_seed,
                read_concurrency=cfg.read_concurrency,
                num_geometry_points=cfg.num_geometry_points,
            )

        def __len__(self) -> int:
            return 1

        def _sample_id(self, idx: int) -> str:
            return "s"

    cfg = ZarrAeroDatasetConfig(
        root=str(tmp_path), split="train", num_surface_points=512, num_volume_points=4096, num_geometry_points=2000
    )
    sample = _DS(cfg)[0]
    assert sample["surface_position"].shape == (512, 3)
    assert sample["volume_velocity"].shape == (4096, 3)
    assert sample["geometry_position"].shape == (2000, 3)


def test_shard_points_cap_splits_arrays_into_multiple_shards(tmp_path: Path) -> None:
    """A shard_points cap packs each array into several chunk-aligned shard objects."""
    writer = ZarrStoreWriter(tmp_path, FILEMAP, "capped", chunk_points=4096, shard_points=8192)
    writer.write_sample("param0/sample", _make_sample())
    writer.save_manifest()

    grid = StoreManifest.load(tmp_path).samples["param0/sample"].domains["volume"]
    assert grid.shard_points == 8192  # cap rounded to a whole number of chunks
    # volume has 28504 points -> ceil(28504 / 8192) = 4 shard objects per array
    shard_objects = sorted((tmp_path / "param0/sample.zarr/volume/velocity/c").rglob("*"))
    assert len([p for p in shard_objects if p.is_file()]) == 4

    # reads are unaffected: full round-trip stays exact, subsampling stays exact-count
    reader = ZarrChunkReader(tmp_path)
    full = reader.read_sample("param0/sample", num_points=None)
    np.testing.assert_array_equal(
        np.sort(full["volume_position"].numpy(), axis=0),
        np.sort(_make_sample()["volume_position"], axis=0),
    )
    assert reader.read_sample("param0/sample", num_points={"volume": 4096})["volume_velocity"].shape == (4096, 3)


def test_shard_points_below_chunk_clamps_to_one_chunk(tmp_path: Path) -> None:
    writer = ZarrStoreWriter(tmp_path, FILEMAP, "tiny-shards", chunk_points=4096, shard_points=1000)
    writer.write_sample("param0/sample", _make_sample())
    grid = writer.manifest.samples["param0/sample"].domains["volume"]
    assert grid.shard_points == 4096  # at least one chunk per shard


def test_manifest_round_trips(tmp_path: Path) -> None:
    writer = ZarrStoreWriter(tmp_path, FILEMAP, "test", chunk_points=4096)
    writer.write_sample("param0/sample", _make_sample())
    writer.save_manifest()
    loaded = StoreManifest.load(tmp_path)
    assert loaded.dataset_name == "test"
    grid = loaded.samples["param0/sample"].domains["volume"]
    assert grid.n_points == NV
    assert grid.chunk_points == 4096
    assert grid.n_chunks == 7


def test_layout_builds_one_array_per_field() -> None:
    layouts = build_domain_layouts(FILEMAP)
    volume = layouts["volume"]
    assert volume.position == "volume_position"
    # One array per present field, in spec order.
    assert list(volume.arrays) == ["volume_position", "volume_velocity", "volume_normals", "volume_sdf"]
    # Positions float32; physical fields float16.
    assert volume.arrays["volume_position"].dtype == "float32"
    assert volume.arrays["volume_velocity"].dtype == "float16"
    assert volume.arrays["volume_velocity"].array_name == "volume/velocity"
    assert volume.arrays["volume_velocity"].dim == 3
    assert volume.arrays["volume_sdf"].dim == 1


def test_selective_field_read(store: Path) -> None:
    """Per-field arrays allow reading a subset of fields without touching the rest."""
    reader = ZarrChunkReader(store)
    out = reader.read_sample(
        "param0/sample", num_points={"volume": 4096}, fields={"volume_position", "volume_velocity"}
    )
    assert set(out) == {"volume_position", "volume_velocity"}
    assert out["volume_position"].shape == (4096, 3)
    assert out["volume_velocity"].shape == (4096, 3)


def test_filename_to_canonical_maps_stored_filenames() -> None:
    mapping = filename_to_canonical(FILEMAP)
    assert mapping["volume_sdf.pt"] == "volume_sdf"
    assert mapping["surface_points.pt"] == "surface_position"


class _StubDataset:
    """Minimal FileMap dataset serving in-memory samples via _load (module-level: picklable)."""

    filemap = FILEMAP
    split = "train"

    def __init__(self, samples: list[dict[str, np.ndarray]]) -> None:
        self._samples = samples
        self._reverse = filename_to_canonical(FILEMAP)  # stored filename -> canonical

    def __len__(self) -> int:
        return len(self._samples)

    def sample_info(self, idx: int) -> dict[str, str]:
        return {"run_name": f"run_{idx}"}

    def _load(self, idx: int, filename: str) -> torch.Tensor:
        return torch.from_numpy(self._samples[idx][self._reverse[filename]])


def _make_stub_dataset(samples: list[dict[str, np.ndarray]]) -> _StubDataset:
    return _StubDataset(samples)


def test_fsspec_source_conversion_streams_pt_files() -> None:
    """convert_fsspec_source discovers and converts .pt samples straight from an fsspec source."""
    import io

    import fsspec

    from noether.data.zarr_store.convert import convert_fsspec_source

    fs = fsspec.filesystem("memory")
    samples = [_make_sample(seed=i) for i in range(2)]
    filenames = {canonical: filename for filename, canonical in filename_to_canonical(FILEMAP).items()}
    for i, sample in enumerate(samples):
        for canonical, filename in filenames.items():
            buffer = io.BytesIO()
            torch.save(torch.from_numpy(sample[canonical]), buffer)
            fs.pipe_file(f"/pt-source-test/preprocessed/run_{i}/{filename}", buffer.getvalue())

    writer = ZarrStoreWriter("memory://pt-source-test/store", FILEMAP, "stream", chunk_points=4096)
    # memory:// is per-process, so stream sequentially (real object stores use max_workers > 1).
    convert_fsspec_source("memory://pt-source-test/preprocessed", FILEMAP, writer, max_workers=1)
    writer.save_manifest()

    reader = ZarrChunkReader("memory://pt-source-test/store")
    assert set(reader.manifest.samples) == {"run_0", "run_1"}
    out = reader.read_sample("run_0", num_points=None)
    np.testing.assert_array_equal(
        np.sort(out["volume_position"].numpy(), axis=0),
        np.sort(samples[0]["volume_position"], axis=0),
    )


def test_filemap_for_dataset_kind_resolves_known_datasets() -> None:
    """The FileMap of any CFD dataset kind is resolvable without instantiating it."""
    from noether.data.datasets.cfd.caeml.filemap import CAEML_FILEMAP
    from noether.data.datasets.cfd.drivaernet.dataset import DrivAerNetDataset
    from noether.data.datasets.cfd.shapenet_car.filemap import SHAPENET_CAR_FILEMAP
    from noether.data.zarr_store.convert import filemap_for_dataset_kind

    assert filemap_for_dataset_kind("noether.data.datasets.cfd.DrivAerNetDataset") is DrivAerNetDataset.FILEMAP
    assert filemap_for_dataset_kind("noether.data.datasets.cfd.ShapeNetCarDataset") is SHAPENET_CAR_FILEMAP
    assert filemap_for_dataset_kind("noether.data.datasets.cfd.DrivAerMLDataset") is CAEML_FILEMAP


def test_float16_overflow_is_rejected_not_stored_as_inf(tmp_path: Path) -> None:
    """Values beyond the target dtype's range must raise, pointing at field_dtypes."""
    sample = _make_sample()
    sample["volume_velocity"] = sample["volume_velocity"] * 1e6  # exceeds float16 max (~6.6e4)
    writer = ZarrStoreWriter(tmp_path, FILEMAP, "overflow", chunk_points=4096)
    with pytest.raises(ValueError, match="exceeds the float16 range.*volume_velocity.*float32"):
        writer.write_group("s", sample)


def test_field_dtype_override_stores_wide_range_fields_losslessly(tmp_path: Path) -> None:
    """A per-field float32 override keeps wide-dynamic-range fields exact."""
    sample = _make_sample()
    sample["volume_velocity"] = sample["volume_velocity"] * 1e6  # vorticity-like magnitudes
    writer = ZarrStoreWriter(tmp_path, FILEMAP, "wide", chunk_points=4096, field_dtypes={"volume_velocity": "float32"})
    writer.write_sample("s", sample)
    writer.save_manifest()

    manifest = StoreManifest.load(tmp_path)
    assert manifest.domains["volume"].arrays["volume_velocity"].dtype == "float32"
    assert manifest.domains["volume"].arrays["volume_normals"].dtype == "float16"  # others unaffected
    out = ZarrChunkReader(tmp_path).read_sample("s", num_points=None)
    np.testing.assert_array_equal(  # float32 storage -> exact round-trip despite huge values
        np.sort(out["volume_velocity"].numpy(), axis=0),
        np.sort(sample["volume_velocity"], axis=0),
    )


def test_broken_sample_is_skipped_not_fatal(tmp_path: Path) -> None:
    """A sample with misaligned fields is skipped with a warning; the rest convert."""
    from noether.data.zarr_store.convert import convert_aero_dataset

    samples = [_make_sample(seed=i) for i in range(3)]
    # Corrupt sample 1: pressure no longer point-aligned with the surface positions.
    samples[1]["surface_pressure"] = np.concatenate([samples[1]["surface_pressure"], np.zeros(100, "float32")])

    writer = ZarrStoreWriter(tmp_path, FILEMAP, "broken", chunk_points=4096)
    convert_aero_dataset(_make_stub_dataset(samples), writer, max_workers=1)
    writer.save_manifest()
    assert set(StoreManifest.load(tmp_path).samples) == {"run_0", "run_2"}


def test_all_samples_failing_raises(tmp_path: Path) -> None:
    from noether.data.zarr_store.convert import convert_aero_dataset

    sample = _make_sample(seed=0)
    sample["surface_pressure"] = sample["surface_pressure"][:100]  # misaligned
    writer = ZarrStoreWriter(tmp_path, FILEMAP, "all-broken", chunk_points=4096)
    with pytest.raises(RuntimeError, match="All 1 samples failed"):
        convert_aero_dataset(_make_stub_dataset([sample]), writer, max_workers=1)


def test_parallel_conversion_matches_sequential(tmp_path: Path) -> None:
    """max_workers > 1 must produce a bit-identical store (per-sample shuffles are seeded)."""
    from noether.data.zarr_store.convert import convert_aero_dataset

    samples = [_make_sample(seed=i) for i in range(5)]
    sequential_root, parallel_root = tmp_path / "seq", tmp_path / "par"

    w_seq = ZarrStoreWriter(sequential_root, FILEMAP, "stub", chunk_points=4096)
    convert_aero_dataset(_make_stub_dataset(samples), w_seq, max_workers=1)
    w_seq.save_manifest()

    w_par = ZarrStoreWriter(parallel_root, FILEMAP, "stub", chunk_points=4096)
    convert_aero_dataset(_make_stub_dataset(samples), w_par, max_workers=4)
    w_par.save_manifest()

    r_seq, r_par = ZarrChunkReader(sequential_root), ZarrChunkReader(parallel_root)
    assert set(r_seq.manifest.samples) == set(r_par.manifest.samples) == {f"run_{i}" for i in range(5)}
    for sample_id in r_seq.manifest.samples:
        a = r_seq.read_sample(sample_id, num_points=None)
        b = r_par.read_sample(sample_id, num_points=None)
        for field in a:
            np.testing.assert_array_equal(a[field].numpy(), b[field].numpy())


def test_store_round_trips_on_fsspec_object_storage() -> None:
    """Writer, manifest and reader all work against an fsspec object store (memory://)."""
    root = "memory://zarr-store-test/store"
    writer = ZarrStoreWriter(root, FILEMAP, "mem", chunk_points=4096)
    writer.write_sample("run_0", _make_sample())
    manifest_path = writer.save_manifest()
    assert manifest_path.startswith("memory://")

    manifest = StoreManifest.load(root)
    assert "run_0" in manifest.samples

    reader = ZarrChunkReader(root)
    sub = reader.read_sample("run_0", num_points={"surface": 1000, "volume": 4096})
    assert sub["surface_position"].shape == (1000, 3)
    assert sub["volume_velocity"].shape == (4096, 3)

    # full read round-trips the float32 positions exactly
    full = reader.read_sample("run_0", num_points=None)
    np.testing.assert_array_equal(
        np.sort(full["volume_position"].numpy(), axis=0),
        np.sort(_make_sample()["volume_position"], axis=0),
    )


def test_convert_aero_dataset_is_dataset_driven(tmp_path: Path) -> None:
    """convert_aero_dataset converts any FileMap dataset via its own filemap/_load/sample ids."""
    from noether.data.zarr_store import filename_to_canonical
    from noether.data.zarr_store.convert import convert_aero_dataset

    reverse = filename_to_canonical(FILEMAP)  # stored filename -> canonical
    samples = [_make_sample(seed=i) for i in range(3)]

    class _StubDataset:
        filemap = FILEMAP
        split = "train"

        def __len__(self) -> int:
            return len(samples)

        def sample_info(self, idx: int) -> dict[str, str]:
            return {"run_name": f"run_{idx}"}

        def _load(self, idx: int, filename: str) -> torch.Tensor:
            return torch.from_numpy(samples[idx][reverse[filename]])

    writer = ZarrStoreWriter(tmp_path, FILEMAP, "stub", chunk_points=4096)
    convert_aero_dataset(_StubDataset(), writer)  # type: ignore[arg-type]
    writer.save_manifest()

    reader = ZarrChunkReader(tmp_path)
    assert set(reader.manifest.samples) == {"run_0", "run_1", "run_2"}  # ids come from sample_info
    out = reader.read_sample("run_1", num_points=None)
    # positions are stored float32 -> exact round-trip (compare sorted; write order is shuffled)
    np.testing.assert_array_equal(
        np.sort(out["volume_position"].numpy(), axis=0),
        np.sort(samples[1]["volume_position"], axis=0),
    )
    assert out["surface_pressure"].shape == (NS,)


def test_store_statistics_match_direct_computation(tmp_path: Path) -> None:
    """Per-field running stats over the store equal a direct float64 computation."""
    from noether.data.zarr_store.statistics import calculate_store_statistics, statistics_to_dict

    writer = ZarrStoreWriter(tmp_path, FILEMAP, "stats", chunk_points=4096)
    s0, s1 = _make_sample(0), _make_sample(1)
    writer.write_sample("a", s0)
    writer.write_sample("b", s1)
    writer.save_manifest()

    stats = calculate_store_statistics(tmp_path)
    assert set(stats) == {
        "surface_position",
        "surface_pressure",
        "surface_normals",
        "volume_position",
        "volume_velocity",
        "volume_sdf",
        "volume_normals",
    }

    for field, st in stats.items():
        # Reference goes through the stored dtype (positions float32, values float16).
        cast = np.float32 if field.endswith("_position") else np.float16
        ref = np.concatenate([s[field].astype(cast).astype(np.float64) for s in (s0, s1)])
        ref = ref.reshape(len(ref), -1)
        np.testing.assert_allclose(st.mean.numpy(), ref.mean(axis=0), rtol=1e-9)
        np.testing.assert_allclose(st.std.numpy(), ref.std(axis=0, ddof=1), rtol=1e-9)
        np.testing.assert_allclose(st.min.numpy(), ref.min(axis=0), rtol=0)
        np.testing.assert_allclose(st.max.numpy(), ref.max(axis=0), rtol=0)
        logref = np.sign(ref) * np.log1p(np.abs(ref))
        np.testing.assert_allclose(st.logmean.numpy(), logref.mean(axis=0), rtol=1e-9)
        assert st.count == len(ref)

    flat = statistics_to_dict(stats)
    pos = np.concatenate([s[f].astype(np.float64) for s in (s0, s1) for f in ("surface_position", "volume_position")])
    np.testing.assert_allclose(flat["raw_pos_min"], [pos.min()], rtol=1e-9)
    np.testing.assert_allclose(flat["raw_pos_max"], [pos.max()], rtol=1e-9)


def test_store_statistics_field_selection_and_workers(store: Path) -> None:
    from noether.data.zarr_store.statistics import calculate_store_statistics

    stats = calculate_store_statistics(store, fields={"surface_pressure", "volume_velocity"}, max_workers=4)
    assert set(stats) == {"surface_pressure", "volume_velocity"}

    stats = calculate_store_statistics(store, exclude_fields={"volume_normals", "surface_normals"})
    assert "volume_normals" not in stats and "surface_normals" not in stats

    with pytest.raises(ValueError, match="Unknown fields"):
        calculate_store_statistics(store, fields={"nope"})
    with pytest.raises(ValueError, match="non-existent"):
        calculate_store_statistics(store, exclude_fields={"nope"})


def test_store_statistics_skips_missing_sample_ids(store: Path) -> None:
    from noether.data.zarr_store.statistics import calculate_store_statistics

    with pytest.warns(UserWarning, match="not in the store"):
        stats = calculate_store_statistics(store, sample_ids=["param0/sample", "missing"])
    assert stats["surface_pressure"].count == NS

    with pytest.raises(ValueError, match="No samples"):
        calculate_store_statistics(store, sample_ids=["missing"])


def test_make_store_selects_local_for_paths(tmp_path: Path) -> None:
    from zarr.storage import LocalStore as _LocalStore

    from noether.data.zarr_store import stores

    assert isinstance(stores.make_store(tmp_path), _LocalStore)
    assert isinstance(stores.make_store(f"file://{tmp_path}"), _LocalStore)


def test_make_store_uses_fsspec_for_non_s3_urls() -> None:
    from zarr.storage import FsspecStore

    from noether.data.zarr_store import stores

    assert isinstance(stores.make_store("memory://some/zarr"), FsspecStore)


def test_make_store_uses_obstore_for_s3_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("obstore")
    from zarr.storage import ObjectStore

    from noether.data.zarr_store import stores

    monkeypatch.setattr(stores, "_obstore_available", lambda: True)
    store = stores.make_store("s3://bucket/prefix.zarr", read_only=True)
    assert isinstance(store, ObjectStore)
    assert store.read_only is True


def test_make_store_falls_back_to_fsspec_for_s3_without_obstore(monkeypatch: pytest.MonkeyPatch) -> None:
    from noether.data.zarr_store import stores

    # Assert routing only: building a real FsspecStore for s3:// would require s3fs.
    monkeypatch.setattr(stores, "_obstore_available", lambda: False)
    seen: dict[str, object] = {}
    monkeypatch.setattr(stores.FsspecStore, "from_url", classmethod(lambda cls, url, **kw: seen.update(url=url, kw=kw)))
    stores.make_store("s3://bucket/prefix.zarr", read_only=True)
    assert seen == {"url": "s3://bucket/prefix.zarr", "kw": {"read_only": True}}

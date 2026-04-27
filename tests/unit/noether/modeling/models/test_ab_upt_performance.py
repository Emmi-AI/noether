#  Copyright © 2026 Emmi AI GmbH. All rights reserved.
"""Benchmarks for the end-to-end AB-UPT forward and forward+backward pass.

Excluded from the default test run — invoke explicitly with ``pytest -m benchmark``.

Exercises the real :class:`AnchoredBranchedUPT` (no mocks) across:

- anchor/geometry token counts (sequence length scaling),
- hidden dimension,
- physics block composition.

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from noether.core.schemas.dataset import DomainDataSpec, FieldDimSpec, ModelDataSpecs
from noether.core.schemas.models import AnchorBranchedUPTConfig
from noether.core.schemas.modules.blocks import TransformerBlockConfig
from noether.core.schemas.modules.encoders import SupernodePoolingConfig
from noether.modeling.models.ab_upt import AnchoredBranchedUPT


def _available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.fixture(params=_available_devices())
def device(request: pytest.FixtureRequest) -> torch.device:
    return torch.device(request.param)


_CACHE_FLUSH_BYTES = 256 * 1024 * 1024  # 256 MiB — larger than typical GPU L2 and CPU L3.


@pytest.fixture
def cache_flush(device: torch.device) -> Callable[[], tuple[tuple, dict]]:
    """Return a ``setup=`` callable that evicts the device cache before each timed iteration.

    Writes a 256 MiB scratch buffer on ``device``, which is large enough to displace any
    hot working-set from L2 (CUDA) or L3 (CPU). Returns ``((), {})`` so it can be passed
    directly as ``setup=`` to :meth:`pytest_benchmark.fixture.BenchmarkFixture.pedantic`.
    """
    buf = torch.empty(_CACHE_FLUSH_BYTES // 4, dtype=torch.float32, device=device)

    def flush() -> tuple[tuple, dict]:
        buf.zero_()
        _sync(device)
        return (), {}

    return flush


def _make_config(
    hidden_dim: int,
    num_heads: int,
    physics_blocks: list[str],
    *,
    decoder_blocks_per_domain: int = 1,
    geometry_depth: int = 1,
) -> AnchorBranchedUPTConfig:
    """Two-domain (``surface`` + ``volume``) config with 3D positions."""
    data_specs = ModelDataSpecs(
        position_dim=3,
        domains={
            "surface": DomainDataSpec(output_dims=FieldDimSpec({"pressure": 1, "shear_stress": 3})),
            "volume": DomainDataSpec(output_dims=FieldDimSpec({"velocity": 3, "pressure": 1})),
        },
    )
    return AnchorBranchedUPTConfig(
        name="ab_upt_bench",
        hidden_dim=hidden_dim,
        geometry_depth=geometry_depth,
        physics_blocks=physics_blocks,
        num_domain_decoder_blocks={"surface": decoder_blocks_per_domain, "volume": decoder_blocks_per_domain},
        data_specs=data_specs,
        supernode_pooling_config=SupernodePoolingConfig(hidden_dim=hidden_dim, input_dim=3, k=4),
        transformer_block_config=TransformerBlockConfig(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_expansion_factor=2,
            use_rope=True,
        ),
    )


def _make_inputs(
    device: torch.device,
    *,
    batch_size: int,
    num_geometry_points: int,
    num_supernodes: int,
    num_surface_anchors: int,
    num_volume_anchors: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a single input batch that matches :meth:`AnchoredBranchedUPT.forward`.

    Supernode indices are the first ``num_supernodes`` points of each sample — enough to satisfy
    the ``k=4`` neighborhood requirement as long as ``num_geometry_points >= 4``.
    """
    assert num_geometry_points >= 4, "need >=4 geometry points per sample for k=4 pooling"
    assert num_supernodes <= num_geometry_points

    gen = torch.Generator(device="cpu").manual_seed(seed)
    total_geom = batch_size * num_geometry_points
    geometry_position = torch.randn(total_geom, 3, generator=gen).to(device)
    geometry_batch_idx = torch.arange(batch_size).repeat_interleave(num_geometry_points).to(device)
    geometry_supernode_idx = torch.cat(
        [torch.arange(num_supernodes) + i * num_geometry_points for i in range(batch_size)]
    ).to(device)

    surface = torch.randn(batch_size, num_surface_anchors, 3, generator=gen).to(device)
    volume = torch.randn(batch_size, num_volume_anchors, 3, generator=gen).to(device)

    return {
        "geometry_position": geometry_position,
        "geometry_supernode_idx": geometry_supernode_idx,
        "geometry_batch_idx": geometry_batch_idx,
        "domain_anchor_positions": {"surface": surface, "volume": volume},
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _forward_closure(model: AnchoredBranchedUPT, inputs: dict[str, Any], device: torch.device) -> Callable[[], None]:
    def step() -> None:
        with torch.inference_mode():
            model(**inputs)
        _sync(device)

    return step


def _forward_backward_closure(
    model: AnchoredBranchedUPT, inputs: dict[str, Any], device: torch.device
) -> Callable[[], None]:
    def step() -> None:
        for p in model.parameters():
            p.grad = None
        predictions, _ = model(**inputs)
        loss = torch.stack([t.sum() for t in predictions.values()]).sum()
        loss.backward()
        _sync(device)

    return step


# --- Parametrization grids ---------------------------------------------------

SEQLEN_GRID = [128, 512, 2048]
HIDDEN_DIM_GRID = [64, 128, 256]
PHYSICS_BLOCKS_GRID: dict[str, list[str]] = {
    "perceiver_self": ["perceiver", "self"],
    "perceiver_joint": ["perceiver", "joint"],
}


# --- Benchmarks --------------------------------------------------------------


# Pedantic-mode knobs used by every benchmark. ``rounds`` timings are reported;
# ``warmup_rounds`` run first but are not measured. The cache flusher is hooked
# in via ``setup=`` so it runs between iterations without polluting the timing.
_ROUNDS = 10
_WARMUP_ROUNDS = 2


@pytest.mark.benchmark(group="ab_upt_seqlen_forward")
@pytest.mark.parametrize("num_anchors", SEQLEN_GRID)
def test_ab_upt_forward_vs_seqlen(benchmark, device, cache_flush, num_anchors):
    """Forward-only — how does runtime scale with per-domain anchor count?

    Fixed: ``hidden_dim=128``, ``num_heads=4``, ``physics_blocks=[perceiver, self]``.
    """
    torch.manual_seed(0)
    config = _make_config(hidden_dim=128, num_heads=4, physics_blocks=["perceiver", "self"])
    model = AnchoredBranchedUPT(config).to(device).eval()
    inputs = _make_inputs(
        device,
        batch_size=1,
        num_geometry_points=num_anchors,
        num_supernodes=max(num_anchors // 4, 4),
        num_surface_anchors=num_anchors,
        num_volume_anchors=num_anchors,
    )
    step = _forward_closure(model, inputs, device)
    benchmark.pedantic(step, setup=cache_flush, rounds=_ROUNDS, warmup_rounds=_WARMUP_ROUNDS, iterations=1)


@pytest.mark.benchmark(group="ab_upt_seqlen_forward_backward")
@pytest.mark.parametrize("num_anchors", SEQLEN_GRID)
def test_ab_upt_forward_backward_vs_seqlen(benchmark, device, cache_flush, num_anchors):
    """Forward+backward — training-time cost scaling with per-domain anchor count.

    Fixed: ``hidden_dim=128``, ``num_heads=4``, ``physics_blocks=[perceiver, self]``.
    """
    torch.manual_seed(0)
    config = _make_config(hidden_dim=128, num_heads=4, physics_blocks=["perceiver", "self"])
    model = AnchoredBranchedUPT(config).to(device).train()
    inputs = _make_inputs(
        device,
        batch_size=1,
        num_geometry_points=num_anchors,
        num_supernodes=max(num_anchors // 4, 4),
        num_surface_anchors=num_anchors,
        num_volume_anchors=num_anchors,
    )
    step = _forward_backward_closure(model, inputs, device)
    benchmark.pedantic(step, setup=cache_flush, rounds=_ROUNDS, warmup_rounds=_WARMUP_ROUNDS, iterations=1)


@pytest.mark.benchmark(group="ab_upt_hidden_dim_forward_backward")
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIM_GRID)
def test_ab_upt_forward_backward_vs_hidden_dim(benchmark, device, cache_flush, hidden_dim):
    """Forward+backward — cost scaling with hidden dimension.

    Fixed: ``num_heads=4``, ``physics_blocks=[perceiver, self]``, 512 anchors per domain.
    """
    torch.manual_seed(0)
    config = _make_config(hidden_dim=hidden_dim, num_heads=4, physics_blocks=["perceiver", "self"])
    model = AnchoredBranchedUPT(config).to(device).train()
    inputs = _make_inputs(
        device,
        batch_size=1,
        num_geometry_points=512,
        num_supernodes=128,
        num_surface_anchors=512,
        num_volume_anchors=512,
    )
    step = _forward_backward_closure(model, inputs, device)
    benchmark.pedantic(step, setup=cache_flush, rounds=_ROUNDS, warmup_rounds=_WARMUP_ROUNDS, iterations=1)


@pytest.mark.benchmark(group="ab_upt_physics_blocks_forward_backward")
@pytest.mark.parametrize("blocks_label", list(PHYSICS_BLOCKS_GRID))
def test_ab_upt_forward_backward_vs_physics_blocks(benchmark, device, cache_flush, blocks_label):
    """Forward+backward — relative cost of each physics-block variant.

    Fixed: ``hidden_dim=128``, ``num_heads=4``, 512 anchors per domain.
    Covers the shared (``self``/``joint``) and per-domain variants in turn.
    """
    torch.manual_seed(0)
    config = _make_config(hidden_dim=128, num_heads=4, physics_blocks=PHYSICS_BLOCKS_GRID[blocks_label])
    model = AnchoredBranchedUPT(config).to(device).train()
    inputs = _make_inputs(
        device,
        batch_size=1,
        num_geometry_points=512,
        num_supernodes=128,
        num_surface_anchors=512,
        num_volume_anchors=512,
    )
    step = _forward_backward_closure(model, inputs, device)
    benchmark.pedantic(step, setup=cache_flush, rounds=_ROUNDS, warmup_rounds=_WARMUP_ROUNDS, iterations=1)

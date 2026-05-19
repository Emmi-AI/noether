#  Copyright © 2026 Emmi AI GmbH. All rights reserved.
"""Benchmarks for :class:`UntiedLinear`'s internal forward strategies.

Excluded from the default test run — invoke explicitly with ``pytest -m benchmark``.

:class:`UntiedLinear` auto-selects the fastest strategy at runtime.
These benchmarks force each strategy individually so we can track their
relative performance across sequence lengths, feature dims, and type counts.

Strategies:

- **split_loop**: ``split`` + one ``F.linear`` per domain (Python loop).
- **equal_size_bmm**: reshape ``(B, S, D)`` → ``(T, B*n, D)`` + single ``torch.bmm``.
  Only valid when all domain sizes are equal.
- **padded_bmm**: pad each chunk to ``max(sizes)``, one ``torch.bmm``, unpad.
  General but wastes FLOPs proportional to padding.
- **grouped_mm**: :func:`torch.nn.functional.grouped_mm` (CUDA only).
  Skipped when unavailable (e.g. CPU or unsupported dtype).
"""

from collections.abc import Sequence

import pytest
import torch

from noether.core.schemas.modules.attention import TokenSpec
from noether.modeling.modules.layers.linear_projection import LinearProjectionConfig
from noether.modeling.modules.untied import UntiedLinear, UntiedLinearConfig, _domain_group_sizes


def _make_layer(num_types: int, dim: int, *, bias: bool = True) -> UntiedLinear:
    """Build an UntiedLinear with ``dim -> dim``."""
    cfg = UntiedLinearConfig(
        num_types=num_types,
        linear_projection=LinearProjectionConfig(input_dim=dim, output_dim=dim, bias=bias, init_weights="torch"),
    )
    return UntiedLinear(cfg)


def _specs_for(seq_len: int, num_types: int, distribution: str) -> Sequence[TokenSpec]:
    """Build token specs totaling ``seq_len`` tokens across ``num_types`` domains.

    ``balanced``: every domain gets roughly ``seq_len / num_types`` tokens.
    ``skewed``: domain 0 gets ~80%, remaining domains share the rest equally.
    """
    if distribution == "balanced":
        base, rem = divmod(seq_len, num_types)
        sizes = [base + (1 if i < rem else 0) for i in range(num_types)]
    elif distribution == "skewed":
        dominant = int(seq_len * 0.8)
        tail = seq_len - dominant
        tail_base, tail_rem = divmod(tail, max(num_types - 1, 1))
        sizes = [dominant] + [tail_base + (1 if i < tail_rem else 0) for i in range(num_types - 1)]
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    assert sum(sizes) == seq_len
    return [TokenSpec(name=f"t{i}", size=s) for i, s in enumerate(sizes)]


def _run_strategy(
    layer: UntiedLinear,
    x: torch.Tensor,
    token_specs: Sequence[TokenSpec],
    strategy: str,
) -> torch.Tensor:
    """Dispatch to a specific forward strategy (or ``auto`` for the default)."""
    sizes = _domain_group_sizes(token_specs)
    B, S, D_in = x.shape
    D_out = layer.output_dim
    T = len(sizes)

    if strategy == "split_loop":
        return layer._split_loop_forward(x, sizes, D_out)
    elif strategy == "equal_size_bmm":
        assert all(s == sizes[0] for s in sizes), "equal_size_bmm requires balanced sizes"
        return layer._equal_size_bmm_forward(x, sizes[0], B, S, D_in, D_out, T)
    elif strategy == "padded_bmm":
        return layer._padded_bmm_forward(x, sizes, B, S, D_in, D_out, T, max(sizes))
    elif strategy == "grouped_mm":
        try:
            return layer._grouped_mm_forward(x, sizes, B, S, D_in, D_out, T)
        except (RuntimeError, NotImplementedError, ValueError) as e:
            pytest.skip(f"grouped_mm unavailable: {e}")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# --- Parametrization grids ---------------------------------------------------

SEQLEN_GRID = [256, 1024, 4096, 16384]
DIM_GRID = [64, 256, 512, 1024]
NUM_TYPES_GRID = [2, 4, 8, 16]
DIST_GRID = ["balanced", "skewed"]
BIAS_GRID = [True, False]
DTYPE_GRID = [torch.float32, torch.bfloat16, torch.float16]
BALANCED_STRATEGIES = ["split_loop", "equal_size_bmm", "grouped_mm"]
ALL_STRATEGIES = ["split_loop", "padded_bmm", "grouped_mm"]


# --- Benchmarks ---------------------------------------------------------------


def _available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.fixture(params=_available_devices())
def device(request: pytest.FixtureRequest) -> torch.device:
    return torch.device(request.param)


@pytest.mark.benchmark(group="untied_linear_seqlen")
@pytest.mark.parametrize("strategy", BALANCED_STRATEGIES)
@pytest.mark.parametrize("seq_len", SEQLEN_GRID)
def test_untied_linear_vs_seqlen(benchmark, device, strategy, seq_len):
    """How does runtime scale with sequence length?

    Fixed: ``dim=256``, ``num_types=4``, balanced distribution.
    """
    dim, num_types = 256, 4
    torch.manual_seed(0)
    layer = _make_layer(num_types, dim).to(device).eval()
    specs = _specs_for(seq_len, num_types, "balanced")
    x = torch.randn(1, seq_len, dim, device=device)

    with torch.inference_mode():
        _run_strategy(layer, x, specs, strategy)
        benchmark(lambda: _run_strategy(layer, x, specs, strategy))


@pytest.mark.benchmark(group="untied_linear_dim")
@pytest.mark.parametrize("strategy", BALANCED_STRATEGIES)
@pytest.mark.parametrize("dim", DIM_GRID)
def test_untied_linear_vs_dim(benchmark, device, strategy, dim):
    """How does runtime scale with the feature dimension?

    Fixed: ``seq_len=2048``, ``num_types=4``, balanced distribution.
    """
    seq_len, num_types = 2048, 4
    torch.manual_seed(0)
    layer = _make_layer(num_types, dim).to(device).eval()
    specs = _specs_for(seq_len, num_types, "balanced")
    x = torch.randn(1, seq_len, dim, device=device)

    with torch.inference_mode():
        _run_strategy(layer, x, specs, strategy)
        benchmark(lambda: _run_strategy(layer, x, specs, strategy))


@pytest.mark.benchmark(group="untied_linear_num_types")
@pytest.mark.parametrize("strategy", BALANCED_STRATEGIES)
@pytest.mark.parametrize("num_types", NUM_TYPES_GRID)
def test_untied_linear_vs_num_types(benchmark, device, strategy, num_types):
    """How does runtime scale with the number of token types?

    Fixed: ``seq_len=2048``, ``dim=256``, balanced distribution.
    """
    seq_len, dim = 2048, 256
    torch.manual_seed(0)
    layer = _make_layer(num_types, dim).to(device).eval()
    specs = _specs_for(seq_len, num_types, "balanced")
    x = torch.randn(1, seq_len, dim, device=device)

    with torch.inference_mode():
        _run_strategy(layer, x, specs, strategy)
        benchmark(lambda: _run_strategy(layer, x, specs, strategy))


@pytest.mark.benchmark(group="untied_linear_distribution")
@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
@pytest.mark.parametrize("distribution", DIST_GRID)
def test_untied_linear_vs_distribution(benchmark, device, strategy, distribution):
    """How sensitive is each strategy to a skewed distribution?

    Fixed: ``seq_len=2048``, ``dim=256``, ``num_types=4``.
    Note: ``equal_size_bmm`` cannot run on skewed distributions.
    """
    seq_len, dim, num_types = 2048, 256, 4
    torch.manual_seed(0)
    layer = _make_layer(num_types, dim).to(device).eval()
    specs = _specs_for(seq_len, num_types, distribution)
    x = torch.randn(1, seq_len, dim, device=device)

    with torch.inference_mode():
        _run_strategy(layer, x, specs, strategy)
        benchmark(lambda: _run_strategy(layer, x, specs, strategy))


@pytest.mark.benchmark(group="untied_linear_bias")
@pytest.mark.parametrize("strategy", BALANCED_STRATEGIES)
@pytest.mark.parametrize("bias", BIAS_GRID)
def test_untied_linear_vs_bias(benchmark, device, strategy, bias):
    """How does the bias term affect runtime?

    Fixed: ``seq_len=2048``, ``dim=256``, ``num_types=4``, balanced distribution.
    """
    seq_len, dim, num_types = 2048, 256, 4
    torch.manual_seed(0)
    layer = _make_layer(num_types, dim, bias=bias).to(device).eval()
    specs = _specs_for(seq_len, num_types, "balanced")
    x = torch.randn(1, seq_len, dim, device=device)

    with torch.inference_mode():
        _run_strategy(layer, x, specs, strategy)
        benchmark(lambda: _run_strategy(layer, x, specs, strategy))


@pytest.mark.benchmark(group="untied_linear_dtype")
@pytest.mark.parametrize("strategy", BALANCED_STRATEGIES)
@pytest.mark.parametrize("dtype", DTYPE_GRID, ids=lambda d: d.__repr__().split(".")[-1])
def test_untied_linear_vs_dtype(benchmark, device, strategy, dtype):
    """How does runtime vary across data types?

    Fixed: ``seq_len=2048``, ``dim=256``, ``num_types=4``, balanced distribution.
    """
    seq_len, dim, num_types = 2048, 256, 4
    torch.manual_seed(0)
    layer = _make_layer(num_types, dim, bias=False).to(device=device, dtype=dtype).eval()
    specs = _specs_for(seq_len, num_types, "balanced")
    x = torch.randn(1, seq_len, dim, device=device, dtype=dtype)

    with torch.inference_mode():
        _run_strategy(layer, x, specs, strategy)
        benchmark(lambda: _run_strategy(layer, x, specs, strategy))


@pytest.mark.benchmark(group="untied_linear_backward")
@pytest.mark.parametrize("strategy", BALANCED_STRATEGIES)
@pytest.mark.parametrize("seq_len", [1024, 4096])
@pytest.mark.parametrize("num_types", [2, 8])
def test_untied_linear_forward_backward(benchmark, device, strategy, seq_len, num_types):
    """Forward+backward — relevant for training-time cost."""
    dim = 256
    torch.manual_seed(0)
    layer = _make_layer(num_types, dim).to(device).train()
    specs = _specs_for(seq_len, num_types, "balanced")
    x = torch.randn(1, seq_len, dim, device=device, requires_grad=True)

    def step() -> None:
        for p in layer.parameters():
            p.grad = None
        out = _run_strategy(layer, x, specs, strategy)
        out.sum().backward()

    step()
    benchmark(step)

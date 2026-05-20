#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Any

from noether.modeling.modules import (
    DeepPerceiverDecoderConfig,
    PerceiverBlockConfig,
    SupernodePoolingConfig,
    TransformerBlockConfig,
)


def _apply_abupt_defaults(data_specs: Any, model_params: dict[str, Any]) -> None:
    """Fill in AB-UPT sub-configs from top-level knobs.

    Derives ``supernode_pooling_config``, ``transformer_block_config``, ``name``, and ``num_domain_decoder_blocks``
    so the user only needs to specify ``hidden_dim``, ``geometry_depth``, ``physics_blocks``, etc.
    """
    hidden_dim = model_params.get("hidden_dim", 192)
    num_heads = model_params.pop("num_heads", 3)
    mlp_expansion_factor = model_params.pop("mlp_expansion_factor", 4)
    use_rope = model_params.pop("use_rope", True)
    radius = model_params.pop("radius", 9)

    model_params.setdefault("name", "ab_upt")
    model_params.setdefault(
        "num_domain_decoder_blocks",
        dict.fromkeys(data_specs.domains, 12),
    )
    model_params.setdefault(
        "supernode_pooling_config",
        SupernodePoolingConfig(
            input_dim=data_specs.position_dim,
            hidden_dim=hidden_dim,
            radius=radius,
        ),
    )
    model_params.setdefault(
        "transformer_block_config",
        TransformerBlockConfig(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_expansion_factor=mlp_expansion_factor,
            use_rope=use_rope,
        ),
    )


def _apply_upt_defaults(data_specs: Any, model_params: dict[str, Any]) -> None:
    """Fill in UPT sub-configs from top-level knobs.

    Derives ``supernode_pooling_config``, ``approximator_config``, and ``decoder_config`` so the user only needs
    to specify ``hidden_dim``, ``num_heads``, ``approximator_depth``, etc.
    """
    hidden_dim = model_params.get("hidden_dim", 192)
    num_heads = model_params.get("num_heads", 3)
    mlp_expansion_factor = model_params.get("mlp_expansion_factor", 4)
    use_rope = model_params.get("use_rope", True)
    radius = model_params.pop("radius", 9)
    decoder_depth = model_params.pop("decoder_depth", 12)

    model_params.setdefault("name", "upt")
    model_params.setdefault("num_heads", num_heads)
    model_params.setdefault("mlp_expansion_factor", mlp_expansion_factor)
    model_params.setdefault("approximator_depth", 12)
    model_params.setdefault("use_rope", use_rope)
    model_params.setdefault("bias_layers", False)
    model_params.setdefault(
        "supernode_pooling_config",
        SupernodePoolingConfig(
            input_dim=data_specs.position_dim,
            hidden_dim=hidden_dim,
            radius=radius,
        ),
    )
    model_params.setdefault(
        "approximator_config",
        TransformerBlockConfig(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_expansion_factor=mlp_expansion_factor,
            use_rope=use_rope,
        ),
    )
    model_params.setdefault(
        "decoder_config",
        DeepPerceiverDecoderConfig(
            perceiver_block_config=PerceiverBlockConfig(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_expansion_factor=mlp_expansion_factor,
                use_rope=use_rope,
            ),
            depth=decoder_depth,
            input_dim=data_specs.position_dim,
        ),
    )


def _apply_transformer_defaults(_data_specs: Any, model_params: dict[str, Any]) -> None:
    """Fill in Transformer sub-configs from top-level knobs.

    Derives ``transformer_block_config`` so the user only needs to specify ``hidden_dim``, ``num_heads``, ``depth``,
    etc.
    """
    hidden_dim = model_params.get("hidden_dim", 192)
    num_heads = model_params.pop("num_heads", 3)
    mlp_expansion_factor = model_params.pop("mlp_expansion_factor", 4)

    model_params.setdefault("name", "transformer")
    model_params.setdefault("depth", 12)
    model_params.setdefault(
        "transformer_block_config",
        TransformerBlockConfig(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_expansion_factor=mlp_expansion_factor,
        ),
    )


def _apply_transolver_defaults(_data_specs: Any, model_params: dict[str, Any]) -> None:
    """Fill in Transolver sub-configs from top-level knobs.

    Same as Transformer defaults, but also sets ``attention_constructor`` and ``attention_arguments`` on the
    ``transformer_block_config``.
    """
    hidden_dim = model_params.get("hidden_dim", 192)
    num_heads = model_params.pop("num_heads", 3)
    mlp_expansion_factor = model_params.pop("mlp_expansion_factor", 4)
    attention_constructor = model_params.get("attention_constructor", "transolver")
    num_slices = model_params.pop("num_slices", 512)

    model_params.setdefault("name", "transolver")
    model_params.setdefault("depth", 12)
    model_params.setdefault("attention_arguments", {"num_slices": num_slices})
    model_params.setdefault(
        "transformer_block_config",
        TransformerBlockConfig(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_expansion_factor=mlp_expansion_factor,
            attention_constructor=attention_constructor,
            attention_arguments={"num_slices": num_slices},
        ),
    )


# Registry of model-kind -> callable that applies smart defaults to model_params.
# Each callable has signature (data_specs, model_params) -> None (mutates model_params in place).
MODEL_DEFAULTS: dict[str, Any] = {
    "noether.modeling.models.aerodynamics.AeroABUPT": _apply_abupt_defaults,
    "noether.modeling.models.aerodynamics.AeroUPT": _apply_upt_defaults,
    "noether.modeling.models.aerodynamics.AeroTransformer": _apply_transformer_defaults,
    "noether.modeling.models.aerodynamics.AeroTransolver": _apply_transolver_defaults,
}

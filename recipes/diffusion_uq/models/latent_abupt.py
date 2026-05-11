#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Any

from models.autoencoder_abupt import ABUPTAutoencoderConfig
from pydantic import ConfigDict, Field
from torch import Tensor, nn

from noether.core.factory.utils import class_constructor_from_class_path
from noether.core.models import Model
from noether.core.models.base import ModelBase
from noether.core.models.composite import CompositeModel
from noether.core.providers.path import PathProvider
from noether.core.schemas.models.base import ModelBaseConfig
from noether.core.schemas.modules.blocks import TransformerBlockConfig
from noether.core.schemas.modules.layers import (
    ContinuousSincosEmbeddingConfig,
    LinearProjectionConfig,
    RopeFrequencyConfig,
)
from noether.core.utils.training import UpdateCounter
from noether.data.container import DataContainer
from noether.modeling.modules.blocks import TransformerBlock
from noether.modeling.modules.layers import (
    ContinuousSincosEmbed,
    LinearProjection,
    RopeFrequency,
)


class LatentDenoiserConfig(ModelBaseConfig):
    model_config = ConfigDict(extra="forbid")

    hidden_dim: int = Field(256, ge=1)
    latent_dim: int = Field(256, ge=1)
    num_heads: int = Field(8, ge=1)
    depth: int = Field(6, ge=1)
    mlp_expansion_factor: int = Field(4, ge=1)
    condition_dim: int = Field(1024, ge=1)
    drop_path: float = Field(0.0, ge=0.0, le=1.0)
    use_rope: bool = Field(False)
    """If True, attention blocks use RoPE keyed on supernode 3d positions (requires
    ``supernode_positions`` at forward time). Off by default for backward-compat."""
    num_surface_tokens: int = Field(0, ge=0)
    """Number of leading tokens that are surface anchors (rest are volume).
    When >0, a learned surface/volume type embedding is added to input tokens,
    matching the ABUPT backbone's surface_bias/volume_bias distinction."""
    adaln_zero_std: float | None = Field(None, ge=0.0)
    """If set, init adaLN modulation linear ~N(0, std) (zero bias) so blocks start
    near-identity. None disables (keeps PyTorch default). 0 = strict DiT zero-init."""


class LatentDiffusionModelConfig(ModelBaseConfig):
    model_config = ConfigDict(extra="forbid")

    autoencoder_config: ABUPTAutoencoderConfig | None = None
    denoiser_config: LatentDenoiserConfig


class LatentDenoiser(Model):
    """Transformer denoiser for latent diffusion with geometry conditioning.

    Per-token conditioning from supernode 3D positions (added to input
    embeddings) plus global timestep conditioning via DiT-style scale/shift/gate
    modulation. Optionally applies RoPE keyed on the same 3D positions inside
    attention.
    """

    def __init__(self, model_config: LatentDenoiserConfig, **kwargs):
        super().__init__(model_config=model_config, **kwargs)
        cfg = model_config
        self.use_rope = cfg.use_rope

        self.time_embed = ContinuousSincosEmbed(
            config=ContinuousSincosEmbeddingConfig(hidden_dim=cfg.hidden_dim, input_dim=1)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.condition_dim),
            nn.SiLU(),
            nn.Linear(cfg.condition_dim, cfg.condition_dim),
        )

        self.pos_embed = ContinuousSincosEmbed(
            config=ContinuousSincosEmbeddingConfig(hidden_dim=cfg.hidden_dim, input_dim=3)
        )

        if self.use_rope:
            assert cfg.hidden_dim % cfg.num_heads == 0
            self.rope = RopeFrequency(
                config=RopeFrequencyConfig(
                    hidden_dim=cfg.hidden_dim // cfg.num_heads,
                    input_dim=3,
                    implementation="complex",
                ),
            )

        self.input_norm = nn.LayerNorm(cfg.latent_dim)
        self.input_proj = LinearProjection(
            config=LinearProjectionConfig(input_dim=cfg.latent_dim, output_dim=cfg.hidden_dim)
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config=TransformerBlockConfig(
                        hidden_dim=cfg.hidden_dim,
                        num_heads=cfg.num_heads,
                        mlp_expansion_factor=cfg.mlp_expansion_factor,
                        condition_dim=cfg.condition_dim,
                        drop_path=cfg.drop_path,
                        bias=True,
                        use_rope=cfg.use_rope,
                    ),
                )
                for _ in range(cfg.depth)
            ]
        )

        self.final_norm = nn.LayerNorm(cfg.hidden_dim)
        self.output_proj = LinearProjection(
            config=LinearProjectionConfig(input_dim=cfg.hidden_dim, output_dim=cfg.latent_dim, init_weights="zeros")
        )

        # surface/volume type embedding mirroring the AB-UPT backbone's surface_bias/volume_bias.
        self.num_surface_tokens = cfg.num_surface_tokens
        self.type_embed = None
        if cfg.num_surface_tokens > 0:
            self.type_embed = nn.Embedding(2, cfg.hidden_dim)

        if cfg.adaln_zero_std is not None:
            self._init_adaln_zero(std=cfg.adaln_zero_std)

    def _init_adaln_zero(self, std: float) -> None:
        """DiT adaLN-zero: init modulation linear near-zero so blocks start as identity."""
        for block in self.blocks:
            mod = getattr(block, "modulation", None)
            if mod is None:
                continue
            proj = getattr(mod, "project", None)
            if isinstance(proj, nn.Linear):
                if std > 0:
                    nn.init.normal_(proj.weight, mean=0.0, std=std)
                else:
                    nn.init.zeros_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)

    def forward(
        self,
        noisy_latents: Tensor,
        timestep: Tensor,
        supernode_positions: Tensor | None = None,
        condition: Tensor | None = None,
    ) -> Tensor:
        """Predict noise, velocity, or denoised latent from noisy latent tokens.

        Args:
            noisy_latents: ``(B, n_tokens, latent_dim)``.
            timestep: ``(B,)`` diffusion timestep, sigma, or flow time.
            supernode_positions: ``(B, n_tokens, 3)`` geometry conditioning per token.
            condition: ``(B, condition_dim)`` optional additional global conditioning.

        Returns:
            ``(B, n_tokens, latent_dim)``.
        """
        t_cond = self.time_mlp(self.time_embed(timestep.view(-1, 1)))
        if condition is not None:
            t_cond = t_cond + condition

        x = self.input_proj(self.input_norm(noisy_latents))
        if supernode_positions is not None:
            x = x + self.pos_embed(supernode_positions)
        if self.type_embed is not None:
            import torch

            n_tokens = x.shape[1]
            type_ids = torch.zeros(n_tokens, dtype=torch.long, device=x.device)
            type_ids[self.num_surface_tokens :] = 1
            x = x + self.type_embed(type_ids)

        attn_kwargs: dict = {}
        if self.use_rope:
            assert supernode_positions is not None, "LatentDenoiser with use_rope=True requires supernode_positions"
            attn_kwargs["freqs"] = self.rope(supernode_positions)

        for block in self.blocks:
            x, _ = block(x, attn_kwargs=attn_kwargs, condition=t_cond)

        return self.output_proj(self.final_norm(x))


class LatentDiffusionModel(CompositeModel):
    """Composite model holding a frozen AB-UPT autoencoder and a trainable latent denoiser.

    Training forward passes noisy latents through the denoiser only. At inference
    time the autoencoder is used to decode sampled latents back to fields.
    """

    def __init__(
        self,
        model_config: LatentDiffusionModelConfig,
        update_counter: UpdateCounter | None = None,
        path_provider: PathProvider | None = None,
        data_container: DataContainer | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_config=model_config,
            update_counter=update_counter,
            path_provider=path_provider,
            data_container=data_container,
        )

        self.autoencoder = None
        if model_config.autoencoder_config is not None:
            ae_config = model_config.autoencoder_config
            ae_config.optimizer_config = None
            ae_cls = class_constructor_from_class_path(ae_config.kind)
            self.autoencoder = ae_cls(
                model_config=ae_config,
                is_frozen=True,
                update_counter=update_counter,
                path_provider=path_provider,
                data_container=data_container,
            )

        self.denoiser = LatentDenoiser(
            model_config=model_config.denoiser_config,
            update_counter=update_counter,
            path_provider=path_provider,
            data_container=data_container,
        )

    @property
    def submodels(self) -> dict[str, ModelBase]:
        models: dict[str, ModelBase] = {"denoiser": self.denoiser}
        if self.autoencoder is not None:
            models["autoencoder"] = self.autoencoder
        return models

    def forward(
        self,
        noisy_latents: Tensor,
        timestep: Tensor,
        supernode_positions: Tensor | None = None,
        condition: Tensor | None = None,
    ) -> Tensor:
        """Training forward — denoiser only."""
        return self.denoiser(noisy_latents, timestep, supernode_positions, condition)

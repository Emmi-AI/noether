#  Copyright © 2025 Emmi AI GmbH. All rights reserved.


import torch
import torch.nn as nn

from noether.core.models import Model
from noether.core.schemas.modules.layers import ContinuousSincosEmbeddingConfig, LinearProjectionConfig
from noether.core.schemas.modules.mlp import MLPConfig
from noether.modeling.modules.layers import ContinuousSincosEmbed, LinearProjection
from noether.modeling.modules.mlp import MLP
from tutorial.schemas.models.base_config import TutorialBaseModelConfig


class BaseModel(Model):
    """Base class for all model we use in this tutorial.

    Args:
        Model: Base class for single models.
    """

    def __init__(
        self,
        model_config: TutorialBaseModelConfig,
        **kwargs,
    ):
        """
        Args:
            model_config: Configuration of the model.
        """

        super().__init__(model_config=model_config, **kwargs)

        self.input_dim = model_config.data_specs.position_dim
        self.output_dim = model_config.data_specs.total_output_dim
        self.use_physics_features = model_config.data_specs.use_physics_features
        self.position_projection = model_config.position_projection
        self.name = model_config.name

        if model_config.hidden_dim:
            if self.position_projection == "sincos":
                self.pos_embed = ContinuousSincosEmbed(
                    config=ContinuousSincosEmbeddingConfig(hidden_dim=model_config.hidden_dim, input_dim=3)
                )
            elif self.position_projection == "linear":
                self.pos_embed = LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=3, output_dim=model_config.hidden_dim, init_weights="truncnormal002"
                    )
                )
            else:
                raise ValueError(
                    f"Unknown position projection: {self.position_projection}. Only 'sincos' and 'linear' are supported."
                )

            if model_config.use_bias_layers:
                self.surface_bias = MLP(
                    config=MLPConfig(
                        input_dim=model_config.hidden_dim,
                        hidden_dim=model_config.hidden_dim,
                        output_dim=model_config.hidden_dim,
                    )
                )

                self.volume_bias = MLP(
                    config=MLPConfig(
                        input_dim=model_config.hidden_dim,
                        hidden_dim=model_config.hidden_dim,
                        output_dim=model_config.hidden_dim,
                    )
                )

        if self.use_physics_features:
            self.project_volume_features = None
            self.project_surface_features = None
            if model_config.data_specs.volume_feature_dim_total > 0:
                self.project_volume_features = LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=model_config.data_specs.volume_feature_dim_total,
                        output_dim=model_config.hidden_dim,
                        init_weights="truncnormal002",
                    )
                )
            if model_config.data_specs.surface_feature_dim_total > 0:
                self.project_surface_features = LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=model_config.data_specs.surface_feature_dim_total,
                        output_dim=model_config.hidden_dim,
                        init_weights="truncnormal002",
                    )
                )
            if not self.project_volume_features and not self.project_surface_features:
                raise ValueError("use_physics_features is True, but both surface and volume feature dims are zero.")

        if model_config.use_output_projection:
            # if use_output_projection is True, we assume that the model has an output projection layer.
            self.use_output_projection = True
            self.norm = nn.LayerNorm(model_config.hidden_dim, eps=1e-6)
            self.out = LinearProjection(
                config=LinearProjectionConfig(
                    input_dim=model_config.hidden_dim, output_dim=self.output_dim, init_weights="truncnormal002"
                )
            )

    def output_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Most model implementations will have an output projection layer that maps the last latent vector into the output physics space.
        We have a unified projection layer that can be used in all models.

        Args:
            x: tensor of shape (batch_size, num_points, dim) containing the features for each point.

        Returns:
            tensor of shape (batch_size, num_points, output_dim) containing the projected features into (normalized) physics space.
        """
        if not self.use_output_projection:
            raise ValueError("output_projection called, but use_output_projection is set to False in the model config.")
        return self.out(self.norm(x))

    def surface_and_volume_bias(self, x: torch.Tensor, surface_mask: torch.Tensor) -> torch.Tensor:
        """For some of the models, the surface and volume are concatenated into a single input tesnor (e.g., Pointnet, Transolver, Transformer).
        For AB-UPT, we shared weight for the physics blocks. Hence, we need to indicate which points are surface and which are volume points.
        We do this by applying a bias (i.e., an MLP) to the surface and volume points separately. The surface mask indicates which points are surface points.
        This function only works for tensors where surface and volume points are concatenated along the second dimension (not for AB-UPT).
        Howerver, for other models, self.surface_bias and self.volume_bias can be called directly in the child class.

        Args:
            x: tensor of shape (batch_size, num_points, input_dim) containing the features for each point.
            surface_mask: Boolean tensor of shape (batch_size, num_points) indicating which points are surface points.

        Returns:
            torch.Tensor: biased tensor x of shape (batch_size, num_points, input_dim) where the surface points have been processed by the surface bias and the volume points by the volume bias.
        """
        unbatch = False
        if x.ndim == 2:
            # if we have a single point, we need to add a batch dimension
            unbatch = True
            x = x.unsqueeze(0)

        surface_mask = surface_mask[0]  #
        x_surface = self.surface_bias(x[:, surface_mask.bool(), :])
        x_volume = self.volume_bias(x[:, ~surface_mask.bool(), :])
        x = torch.concat([x_surface, x_volume], dim=1)
        if unbatch:
            x = x.squeeze(0)
        return x

    def gather_outputs(self, x: torch.Tensor, surface_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """The output gathering function is used to extract the relevant outputs from the model's output tensor.
        It assumes that the output tensor has a specific structure, where the first dimension corresponds to the batch size,
        the second dimension corresponds to the surface/volume points, and the third dimension corresponds to the output features

        The surface pressure is expected to be at index 0, the volume velocity at indices 1:4,
        and if the output dimension is 11, the surface wall shear stress is at indices 4:7,
        the volume total pressure coefficient at index 7, and the volume vorticity at indices 8:11. These last three are only available
        for AhmedML and DriverML datasets.

        Args:
            x: output tensor from the model, shape (batch_size, num_points, output_dim)
            surface_mask: Indicator boolean tensor for surface points, shape (batch_size, num_points). All surface points should be True, and all volume points should be False.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the gathered outputs:
                - "surface_pressure": Tensor of shape (batch_size, num_surface_points, 1)
                - "volume_velocity": Tensor of shape (batch_size, num_volume_points, 3)
                - "surface_wallshearstress": Tensor of shape (batch_size, num_surface_points, 3) if output_dim is 11
                - "volume_totalpcoeff": Tensor of shape (batch_size, num_volume_points, 1) if output_dim is 11
                - "volume_vorticity": Tensor of shape (batch_size, num_volume_points, 3) if output_dim is 11
        """
        # assumes surface pressure on index 0, 1:4 volume velocity

        surface_mask = surface_mask[0]  # we assume the surface mask is the same for all samples in the batch
        surface_pressure = x[:, surface_mask.bool(), :1]
        volume_velocity = x[
            :, ~surface_mask.bool(), 1:4
        ]  # when we only have one volume point, don't compute loss (default of zero is not possible)

        extra_out = {}
        if self.output_dim > 4:
            assert self.output_dim == 11
            # dim 0: surface pressure, dim 1:4 volume velocity, dim 4:6 surface_wallshearstress, dim 6:7 volume_totalpcoeff, dim 7: 10 volume_vorticity
            surface_friction = x[:, surface_mask.bool(), 4:7]
            volume_pressure = x[:, ~surface_mask.bool(), 7:8]
            volume_vorticity = x[:, ~surface_mask.bool(), 8:11]

            extra_out["surface_friction"] = surface_friction
            extra_out["volume_pressure"] = volume_pressure
            extra_out["volume_vorticity"] = volume_vorticity

        return {
            "surface_pressure": surface_pressure,
            "volume_velocity": volume_velocity,
            **extra_out,
        }

    def _init_weights(self, module: nn.Module) -> None:
        """private method to initialize the weights of the model.

        Args:
            module: nn.Module to initialize weights for. This is used to initialize the weights of the model.
        """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

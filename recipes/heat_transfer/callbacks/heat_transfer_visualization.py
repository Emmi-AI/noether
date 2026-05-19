#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import torch
from pydantic import Field

from noether.core.callbacks.periodic import PeriodicDataIteratorCallback, PeriodicDataIteratorCallbackConfig

VISUALIZATION_FIELDS = [
    "volume_velocity",
    "volume_temperature",
    "volume_pressure",
]

METRIC_SUFFIX_TARGET = "_target"


class HeatTransferVisualizationCallbackConfig(PeriodicDataIteratorCallbackConfig):
    """Configuration for heat transfer 3D scatter visualization callback."""

    kind: str | None = "heat_transfer.callbacks.HeatTransferVisualizationCallback"

    forward_properties: list[str] = []
    """List of properties in the dataset to be forwarded during inference."""
    batch_size: int = Field(1)
    """Batch size for evaluation. Currently only batch_size=1 is supported."""


class HeatTransferVisualizationCallback(PeriodicDataIteratorCallback):
    """Callback that visualizes ground truth, prediction, and error
    for the first test sample as interactive plotly 3D scatter plots.

    One figure per field (velocity, temperature, pressure) is logged with three
    subplots: ground truth, prediction, and per-point error.
    All values are shown in denormalized (physical) space.
    """

    def __init__(self, callback_config: HeatTransferVisualizationCallbackConfig, **kwargs):
        super().__init__(callback_config, **kwargs)

        self.forward_properties = callback_config.forward_properties

    def process_data(self, batch: dict[str, torch.Tensor], **_) -> dict[str, torch.Tensor]:
        """Run inference and collect positions, predictions, and targets."""
        forward_inputs = {k: v for k, v in batch.items() if k in self.forward_properties}
        with self.trainer.autocast_context:
            model_outputs = self.model(**forward_inputs)

        result = {"positions": batch["volume_anchor_position"]}
        for field in VISUALIZATION_FIELDS:
            prediction = model_outputs.get(field)
            target = batch.get(f"{field}{METRIC_SUFFIX_TARGET}")
            if prediction is not None and target is not None:
                result[f"{field}_pred"] = prediction
                result[f"{field}_target"] = target

        return result

    def process_results(self, results: dict[str, torch.Tensor], **_) -> None:
        """Create and log plotly figures for the first sample."""
        if not results:
            self.logger.warning(f"No results for dataset '{self.dataset_key}'")
            return

        # Take the first sample (index 0 along the batch dimension)
        dataset = self.data_container.get_dataset(self.dataset_key)
        positions = dataset.denormalize("volume_position", results["positions"][:1])[0]

        for field in VISUALIZATION_FIELDS:
            pred_key = f"{field}_pred"
            target_key = f"{field}_target"
            if pred_key not in results or target_key not in results:
                continue

            pred = dataset.denormalize(field, results[pred_key][:1])[0]
            target = dataset.denormalize(field, results[target_key][:1])[0]

            fig = self._create_figure(positions.cpu(), pred.cpu(), target.cpu(), field)
            self.writer.add_nonscalar(
                key=f"visualization/{self.dataset_key}/{field}",
                value=fig,
            )
            cp = self.trainer.update_counter.cur_iteration
            uri = (
                self.checkpoint_writer.path_provider.run_output_path
                / "visualization"
                / self.dataset_key
                / f"{field}_cp={cp}.html"
            )
            uri.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(uri)

        self.logger.debug(f"Logged visualization for dataset '{self.dataset_key}'")

    @staticmethod
    def _create_figure(
        positions: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        field_name: str,
    ):
        """Build a plotly figure with ground truth, prediction, and error subplots."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        x = positions[:, 0].numpy()
        y = positions[:, 1].numpy()
        z = positions[:, 2].numpy()

        # Reduce to scalar: magnitude for vectors, squeeze for scalars
        if predictions.shape[-1] > 1:
            pred_vals = predictions.norm(dim=-1).numpy()
            target_vals = targets.norm(dim=-1).numpy()
            value_label = f"|{field_name}|"
        else:
            pred_vals = predictions.squeeze(-1).numpy()
            target_vals = targets.squeeze(-1).numpy()
            value_label = field_name

        # Per-point error
        errors = (predictions - targets).squeeze()

        if errors.ndim > 1:
            # vector field instead of scalar field -> use projection error
            errors = torch.sum(errors * targets / (torch.norm(targets, dim=-1, keepdim=True)), dim=-1)

        # Shared color range for ground truth and prediction
        vmin = float(min(pred_vals.min(), target_vals.min()))
        vmax = float(max(pred_vals.max(), target_vals.max()))

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=["Ground Truth", "Prediction", "Error"],
            specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
            horizontal_spacing=0.02,
        )

        common_marker = dict(size=1.25, colorscale="RdBu_r", opacity=0.8)

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    **common_marker,
                    color=target_vals,
                    cmin=vmin,
                    cmax=vmax,
                    colorbar=dict(title=value_label, x=0.28, len=0.8),
                ),
                name="Ground Truth",
                hovertemplate="%{marker.color:.1f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(**common_marker, color=pred_vals, cmin=vmin, cmax=vmax, colorbar=None),
                name="Prediction",
                hovertemplate="%{marker.color:.1f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        error_max = torch.max(torch.abs(errors)).item()
        error_max = 1.0 if error_max < 1e-9 else error_max

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=1.25,
                    colorscale="RdBu_r",
                    color=errors.numpy(),
                    cmin=-error_max,
                    cmax=error_max,
                    colorbar=dict(title="Error", x=0.96, len=0.8),
                ),
                hovertemplate="%{marker.color:.1f}<extra></extra>",
                name="Prediction Error",
            ),
            row=1,
            col=3,
        )

        scene = dict(
            xaxis=dict(showbackground=False, showgrid=False),
            yaxis=dict(showbackground=False, showgrid=False),
            zaxis=dict(showbackground=False, showgrid=False),
            camera=dict(eye=dict(x=1, y=0.25, z=0.1), center=dict(x=0, y=0, z=-0.75)),
        )

        # Use consistent camera across subplots
        fig.update_layout(
            title=f"{field_name}",
            height=500,
            width=1500,
            showlegend=False,
            scene=scene,
            scene2=scene,
            scene3=scene,
        )

        return fig

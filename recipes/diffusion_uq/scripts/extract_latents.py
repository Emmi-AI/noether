#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Extract latent tokens from a trained AB-UPT autoencoder.

Usage from notebook::

    extract_latents(config, output_root="./data/latents", device="cuda")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from noether.core.schemas.schema import ConfigSchema
from noether.training.runners import HydraRunner


def _resolve_checkpoint(config: ConfigSchema, checkpoint_path: str | None) -> Path | None:
    """Return explicit checkpoint path if given, else search output dir for the best one."""
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"AE checkpoint not found: {p}")
        return p
    ae_output = Path(str(config.output_path))
    if not ae_output.exists():
        return None
    for run_dir in sorted(ae_output.iterdir(), reverse=True):
        candidate = run_dir / "checkpoints" / f"{config.model.name}_cp=best_model.loss.test.total_model.th"
        if candidate.exists():
            return candidate
    return None


def extract_latents(
    config: ConfigSchema,
    output_root: str = "./data/latents_abupt",
    device: str = "cuda",
    batch_size: int = 1,
    checkpoint_path: str | None = None,
):
    """Extract latents from a trained AB-UPT autoencoder.

    Loads the best checkpoint (or ``checkpoint_path`` if given), runs ``encode()``
    on all splits, and saves per-sample ``.pt`` files plus latent statistics.
    """
    output_root = Path(output_root)

    trainer, model, tracker, _ = HydraRunner.setup_experiment(device=device, config=config)

    best_ckpt = _resolve_checkpoint(config, checkpoint_path)
    if best_ckpt is not None:
        state = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["state_dict"])
        print(f"loaded checkpoint: {best_ckpt}")
    else:
        print("WARNING: no checkpoint found, using random weights")

    model.eval()
    model.to(device)

    data_container = trainer.data_container

    for split_key in config.datasets:
        split_dir = output_root / split_key
        split_dir.mkdir(parents=True, exist_ok=True)

        dataset = data_container.get_dataset(split_key)
        pipeline = dataset.pipeline

        all_latents = []
        all_positions = []
        n_samples = len(dataset)

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                samples = [dataset[i] for i in range(start, end)]
                batch = pipeline(samples)
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                surface_fields = (
                    torch.cat(
                        [batch[k] for k in ["surface_pressure_target", "surface_friction_target"] if k in batch], dim=-1
                    )
                    if model.surface_field_proj is not None
                    else None
                )

                volume_fields = (
                    torch.cat(
                        [
                            batch[k]
                            for k in ["volume_pressure_target", "volume_velocity_target", "volume_vorticity_target"]
                            if k in batch
                        ],
                        dim=-1,
                    )
                    if model.volume_field_proj is not None
                    else None
                )

                latents, super_pos = model.encode(
                    geometry_position=batch["geometry_position"],
                    geometry_supernode_idx=batch["geometry_supernode_idx"],
                    geometry_batch_idx=batch["geometry_batch_idx"],
                    surface_anchor_position=batch["surface_anchor_position"],
                    volume_anchor_position=batch["volume_anchor_position"],
                    surface_fields=surface_fields,
                    volume_fields=volume_fields,
                )

                # perceiver: zeros (positionless); sampled: super_pos; none: anchors.
                if getattr(model, "use_token_bottleneck", False):
                    sp_surf = super_pos.get("surface")
                    sp_vol = super_pos.get("volume")
                    if sp_surf is not None and sp_vol is not None:
                        anchor_pos = torch.cat([sp_surf, sp_vol], dim=1)
                    else:
                        K = latents.shape[1]
                        anchor_pos = torch.zeros(latents.shape[0], K, 3, device=latents.device)
                else:
                    sp_surf = sp_vol = None
                    anchor_pos = torch.cat(
                        [
                            batch["surface_anchor_position"],
                            batch["volume_anchor_position"],
                        ],
                        dim=1,
                    )

                # Slim payload: only the per-sample tensors that the AE encoder
                # produces. Anchor positions, geometry, and ground-truth fields
                # are re-derived at consume-time from the underlying CFD dataset
                # via :class:`LatentDataset`.
                bs = end - start
                for j in range(bs):
                    idx = start + j
                    lat_j = latents[j].cpu()
                    pos_j = anchor_pos[j].cpu()
                    all_latents.append(lat_j)
                    all_positions.append(pos_j)
                    payload = {
                        "latents": lat_j,
                        "supernode_positions": pos_j,
                        # per-branch super-token positions (sampled bottleneck only) — KV pos for decode.
                        "super_position_surface": sp_surf[j].cpu() if sp_surf is not None else None,
                        "super_position_volume": sp_vol[j].cpu() if sp_vol is not None else None,
                    }
                    torch.save(payload, split_dir / f"sample_{idx:05d}.pt")

                if start % 50 == 0:
                    print(f"  [{split_key}] {start}/{n_samples}")

        stacked = torch.stack(all_latents)
        latent_var = stacked.var(dim=(0, 1)).numpy().astype(np.float64)
        latent_scale = 1.0 / float(np.sqrt(np.maximum(np.mean(latent_var), 1e-12)))

        torch.save(
            {
                "latent_mean": stacked.mean(dim=0),
                "latent_std": stacked.std(dim=0),
                "latent_scale": latent_scale,
            },
            output_root / f"{split_key}_stats.pt",
        )
        print(f"  [{split_key}] done — {len(all_latents)} samples, latent_scale={latent_scale:.4f}")

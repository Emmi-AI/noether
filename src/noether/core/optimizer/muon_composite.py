#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""MuonComposite: torch.optim.Muon for 2D params, any optimizer for the rest.

Parameters are routed based on dimensionality: ``ndim >= 2`` goes to Muon,
everything else (biases, norms, 1D embeddings) goes to the secondary optimizer.
"""

import torch


class MuonComposite(torch.optim.Optimizer):
    """Composite optimizer using torch.optim.Muon for 2D weight matrices
    and a configurable secondary optimizer for all other parameters (biases, norms, embeddings).
    """

    def __init__(
        self,
        params,
        lr=0.01,
        momentum=0.95,
        weight_decay=0.01,
        secondary=None,
        nesterov=None,
        ns_steps=None,
        adjust_lr_fn=None,
    ):
        """
        Args:
            params: Iterable of parameter groups.
            lr: Learning rate for the Muon optimizer.
            momentum: Momentum factor for the Muon optimizer.
            weight_decay: Weight decay for the Muon optimizer.
            secondary: Configuration dict for the secondary optimizer (biases, norms, embeddings).
            nesterov: Enable Nesterov momentum in Muon. None uses Muon's default (True).
            ns_steps: Number of Newton-Schulz iteration steps. None uses Muon's default (5).
            adjust_lr_fn: Per-matrix LR adjustment strategy for Muon. One of ``"original"``
                or ``"match_rms_adamw"``. None uses Muon's default (``"original"``).
        """
        params = list(params)

        # Resolve secondary optimizer kwargs: fall back to primary lr/weight_decay if not set.
        secondary_kwargs = dict(secondary) if secondary else {}
        secondary_kind = secondary_kwargs.pop("kind", None) or "noether.core.optimizer.Lion"

        # If the secondary optimizer config doesn't specify lr/wd, it inherits from the primary config
        secondary_kwargs.setdefault("lr", lr)
        secondary_kwargs.setdefault("weight_decay", weight_decay)

        # Route params by dimensionality: 2D+ -> Muon, 1D/0D -> secondary.
        # A single incoming group may be split across both optimizers.
        muon_groups = []
        secondary_groups = []
        for group in params:
            group_params = group["params"]
            other = {k: v for k, v in group.items() if k != "params"}
            muon_params = [p for p in group_params if p.ndim >= 2]
            secondary_params = [p for p in group_params if p.ndim < 2]
            if muon_params:
                muon_groups.append({**other, "params": muon_params})
            if secondary_params:
                secondary_groups.append({**other, "params": secondary_params})

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Create internal Muon optimizer, forwarding only explicitly set kwargs
        muon_kwargs: dict = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        if nesterov is not None:
            muon_kwargs["nesterov"] = nesterov
        if ns_steps is not None:
            muon_kwargs["ns_steps"] = ns_steps
        if adjust_lr_fn is not None:
            muon_kwargs["adjust_lr_fn"] = adjust_lr_fn
        self._muon = torch.optim.Muon(muon_groups, **muon_kwargs) if muon_groups else None
        from noether.core.factory.utils import class_constructor_from_class_path

        secondary_cls = class_constructor_from_class_path(secondary_kind)
        self._secondary = secondary_cls(secondary_groups, **secondary_kwargs) if secondary_groups else None

        # Replace param_groups with references from internal optimizers
        # so that external lr/wd mutations (e.g. from schedulers) propagate directly
        self.param_groups = []
        if self._muon:
            self.param_groups.extend(self._muon.param_groups)
        if self._secondary:
            self.param_groups.extend(self._secondary.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._muon:
            self._muon.step()
        if self._secondary:
            self._secondary.step()
        return loss

    def zero_grad(self, set_to_none=True):
        if self._muon:
            self._muon.zero_grad(set_to_none)
        if self._secondary:
            self._secondary.zero_grad(set_to_none)

    def state_dict(self):
        # Each child optimizer indexes its params starting from 0, so muon and secondary
        # use overlapping keys. Renumber them into a single global counter (`idx`) so the
        # merged state/param_groups don't collide.
        state = {}
        param_groups = []
        idx = 0
        for opt in (self._muon, self._secondary):
            if opt is None:
                continue
            sd = opt.state_dict()
            idx_map = {}
            for group in sd["param_groups"]:
                new_params = []
                for old_idx in group["params"]:
                    idx_map[old_idx] = idx
                    new_params.append(idx)
                    idx += 1
                param_groups.append({**group, "params": new_params})
            for old_idx, s in sd["state"].items():
                state[idx_map[old_idx]] = s
        return {"state": state, "param_groups": param_groups}

    def load_state_dict(self, state_dict):
        # The merged dict from state_dict() uses global param indices, but each child
        # optimizer expects its own local indices starting at 0, hence this function
        # maps the global indices back to local ones for each optimizer.
        sd_state = state_dict["state"]
        sd_groups = state_dict["param_groups"]

        total_groups = sum(len(opt.param_groups) for opt in (self._muon, self._secondary) if opt is not None)
        if len(sd_groups) != total_groups:
            raise ValueError(
                f"MuonComposite.load_state_dict: checkpoint has {len(sd_groups)} param groups but current "
                f"split expects {total_groups}. The 2D/non-2D param routing likely changed (model architecture "
                "or param_group modifiers differ from the saved run)."
            )

        offset = 0
        for opt in (self._muon, self._secondary):
            if opt is None:
                continue
            n_groups = len(opt.param_groups)
            opt_sd_groups = sd_groups[offset : offset + n_groups]

            opt_state = {}
            new_idx = 0
            remapped_groups = []
            for group in opt_sd_groups:
                new_params = []
                for orig_idx in group["params"]:
                    if orig_idx in sd_state:
                        opt_state[new_idx] = sd_state[orig_idx]
                    new_params.append(new_idx)
                    new_idx += 1
                remapped_groups.append({**group, "params": new_params})

            opt.load_state_dict({"state": opt_state, "param_groups": remapped_groups})
            offset += n_groups

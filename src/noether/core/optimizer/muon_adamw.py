#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""MuonAdamW: torch.optim.Muon for 2D params, AdamW for the rest.

Param groups are tagged with use_muon=True/False by OptimizerWrapper
based on parameter dimensionality (ndim >= 2 -> Muon, otherwise -> AdamW).
"""

import torch


class MuonAdamW(torch.optim.Optimizer):
    """Composite optimizer using torch.optim.Muon for 2D weight matrices
    and torch.optim.AdamW for all other parameters (biases, norms, embeddings).

    Config example::

        kind: noether.core.optimizer.MuonAdamW
        lr: 2.0e-2
        momentum: 0.95
        weight_decay: 0.01
    """

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01):
        params = list(params)

        # Split groups by use_muon flag before super().__init__ modifies them
        muon_groups = []
        adam_groups = []
        for group in params:
            clean = {k: v for k, v in group.items() if k != "use_muon"}
            if group.get("use_muon", True):
                muon_groups.append(clean)
            else:
                adam_groups.append(clean)

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Create internal optimizers
        self._muon = (
            torch.optim.Muon(muon_groups, lr=lr, momentum=momentum, weight_decay=weight_decay) if muon_groups else None
        )
        self._adamw = torch.optim.AdamW(adam_groups, lr=lr, weight_decay=weight_decay) if adam_groups else None

        # Replace param_groups with references from internal optimizers
        # so that external lr/wd mutations (e.g. from schedulers) propagate directly
        self.param_groups = []
        if self._muon:
            self.param_groups.extend(self._muon.param_groups)
        if self._adamw:
            self.param_groups.extend(self._adamw.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._muon:
            self._muon.step()
        if self._adamw:
            self._adamw.step()
        return loss

    def zero_grad(self, set_to_none=True):
        if self._muon:
            self._muon.zero_grad(set_to_none)
        if self._adamw:
            self._adamw.zero_grad(set_to_none)

    def state_dict(self):
        state = {}
        param_groups = []
        idx = 0
        for opt in (self._muon, self._adamw):
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
        sd_state = state_dict["state"]
        sd_groups = state_dict["param_groups"]

        offset = 0
        for opt in (self._muon, self._adamw):
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

#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set the seed for random number generation for Python `random`, numpy,
    torch and torch.cuda, if available.

    Args:
        seed: Seed value.
    """
    logger.info(f"Seeding process RNG with seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that re-seeds Python ``random`` and ``numpy``
    inside each worker.

    PyTorch already derives a per-worker torch seed from the ``DataLoader``'s
    ``generator`` (or ``base_seed``) combined with ``worker_id``, so
    ``torch.randperm``, ``torch.randn``, etc. inside workers are deterministic
    as long as the ``DataLoader`` receives a seeded ``generator``. However,
    ``random`` and ``numpy.random`` are forked from the main process unseeded
    per-worker, which makes any code using them in a worker non-deterministic
    across runs. This function pulls the torch worker seed and uses it to
    reseed ``random`` and ``numpy`` so the whole worker is deterministic.

    Args:
        worker_id: Worker id passed by the DataLoader. Unused; kept to match
            the ``worker_init_fn`` signature.
    """
    del worker_id  # torch.initial_seed() already encodes the worker id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

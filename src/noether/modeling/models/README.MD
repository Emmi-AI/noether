# `Noether` model zoo

The `Noether Framework` includes base implementations for several state-of-the-art models:

| Model | Paper | Implementation | Notes
|-------|-------|----------------|-------|
| **AB-UPT** | [arXiv:2502.09692](https://arxiv.org/abs/2502.09692) | [ab_upt.py](./ab_upt.py) | - |
| **Transformer** | - | [transformer.py](./transformer.py) | - |
| **Transolver** | [arXiv:2402.02366](https://arxiv.org/abs/2402.02366) | [transolver.py](./transolver.py) | Transolver is a Transformer with a different attention [config](../../core/schemas/models/transolver.py) |
| **Transolver++** | [arXiv:2502.02414](https://arxiv.org/abs/2502.02414) | Schema only: [transolver.py](../../core/schemas/models/transolver.py#21) | | Transolver++ is a Transolver with a different attention config |
| **UPT** | [arXiv:2402.12365](https://arxiv.org/abs/2402.12365) | [upt.py](./model/upt.py) | - |
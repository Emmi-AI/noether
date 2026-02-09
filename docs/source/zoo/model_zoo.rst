Noether Model Zoo
=================

The Noether Framework includes base implementations for several state-of-the-art models:

.. list-table::
   :header-rows: 1
   :widths: 15 20 30 35

   * - Model
     - Paper
     - Implementation
     - Notes
   * - **AB-UPT**
     - `arXiv:2502.09692 <https://arxiv.org/abs/2502.09692>`_
     - `ab_upt.py <https://github.com/Emmi-AI/noether/blob/main/src/noether/modeling/models/ab_upt.py>`_
     - \-
   * - **Transformer**
     - \-
     - `transformer.py <https://github.com/Emmi-AI/noether/blob/main/src/noether/modeling/models/transformer.py>`_
     - \-
   * - **Transolver**
     - `arXiv:2402.02366 <https://arxiv.org/abs/2402.02366>`_
     - `transolver.py <https://github.com/Emmi-AI/noether/blob/main/src/noether/modeling/models/transolver.py>`_
     - Transolver is a Transformer with a different attention `config <https://github.com/Emmi-AI/noether/tree/main/src/noether/core/schemas/models/transolver.py>`_
   * - **Transolver++**
     - `arXiv:2502.02414 <https://arxiv.org/abs/2502.02414>`_
     - Schema only: `transolver.py <https://github.com/Emmi-AI/noether/tree/main/src/noether/core/schemas/models/transolver.py#21>`_
     - Transolver++ is a Transolver with a different attention config
   * - **UPT**
     - `arXiv:2402.12365 <https://arxiv.org/abs/2402.12365>`_
     - `upt.py <https://github.com/Emmi-AI/noether/blob/main/src/noether/modeling/models/upt.py>`_
     - \-

- Transformer & Transolver(++): These models are implemented as a backbone consisting of a stack of layers. Transolver replaces the standard attention mechanism with Physics-Attention. For these models, input embedding and output projection must be handled by separate wrapper modules.
- UPT & AB-UPT: In contrast, these models are "off-the-shelf" implementations.``
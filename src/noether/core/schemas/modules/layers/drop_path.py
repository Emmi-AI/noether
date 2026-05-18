#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-export for ``UnquantizedDropPathConfig``.

The config has moved next to its matching class in
:mod:`noether.modeling.modules.layers.drop_path`.
"""

from noether.modeling.modules.layers.drop_path import UnquantizedDropPathConfig

__all__ = ["UnquantizedDropPathConfig"]

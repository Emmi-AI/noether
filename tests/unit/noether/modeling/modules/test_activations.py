#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch.nn as nn

from noether.modeling.modules.activations import Activation


def test_build_creates_new_instances():
    """Each call to build() should return a distinct nn.Module instance."""
    a1 = Activation.GELU.build()
    a2 = Activation.GELU.build()
    assert isinstance(a1, nn.GELU)
    assert isinstance(a2, nn.GELU)
    assert a1 is not a2


def test_build_all_activations():
    """Every Activation enum member should produce a valid nn.Module."""
    for act in Activation:
        module = act.build()
        assert isinstance(module, nn.Module)


def test_value_is_class_not_instance():
    """Enum values should be classes, not instances."""
    for act in Activation:
        assert isinstance(act.value, type)

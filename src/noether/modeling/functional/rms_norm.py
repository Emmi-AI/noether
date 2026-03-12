#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import torch.nn.functional as F


def norm(x):
    """Returns the RMS norm of the input tensor `x` along the last dimension."""
    return F.rms_norm(x, (x.size(-1),))

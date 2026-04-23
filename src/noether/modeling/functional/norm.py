#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import torch.nn.functional as F


def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))

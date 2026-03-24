#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from .aero_cfd import AeroCFDPreset
from .ahmedml import AhmedMLPreset
from .drivaerml import DrivAerMLPreset
from .drivaernet import DrivAerNetPreset
from .emmi_wing import EmmiWingPreset
from .shapenet_car import ShapeNetCarPreset

__all__ = [
    "AeroCFDPreset",
    "AhmedMLPreset",
    "DrivAerMLPreset",
    "DrivAerNetPreset",
    "EmmiWingPreset",
    "ShapeNetCarPreset",
]

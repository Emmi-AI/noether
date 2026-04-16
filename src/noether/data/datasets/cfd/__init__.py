#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from .caeml.ahmedml import AhmedMLDataset, AhmedMLDefaultSplitIDs
from .caeml.drivaerml import DrivAerMLDataset, DrivAerMLDefaultSplitIDs
from .drivaernet.dataset import DrivAerNetDataset
from .emmi_wing import EmmiWingDataset, EmmiWingHFDataset
from .shapenet_car import ShapeNetCarDataset, ShapeNetCarDefaultSplitIDs
from .simshift_heatsink import SimshiftHeatsinkDataset

__all__ = [
    "AhmedMLDataset",
    "AhmedMLDefaultSplitIDs",
    "DrivAerMLDataset",
    "DrivAerMLDefaultSplitIDs",
    "DrivAerNetDataset",
    "EmmiWingDataset",
    "EmmiWingHFDataset",
    "ShapeNetCarDataset",
    "ShapeNetCarDefaultSplitIDs",
    "SimshiftHeatsinkDataset",
]

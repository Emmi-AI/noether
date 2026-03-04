#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import math
from itertools import groupby
from typing import Any

import numpy as np
import torch

_SI_SUFFIXES_LARGE = ["", "K", "M", "G", "T", "P", "E"]
_SI_SUFFIXES_SMALL = ["m", "µ", "n", "p", "f", "a"]  # m = 10^-3, µ = 10^-6, etc.


class CustomFormatter(logging.Formatter):
    # Define ANSI escape codes for colors
    GREY = "\x1b[38;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    def __init__(self, fmt: str, datefmt: str | None = None, colors: bool = True):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.GREY + self.fmt + self.RESET,
            logging.INFO: self.fmt,
            logging.WARNING: self.YELLOW + self.fmt + self.RESET,
            logging.ERROR: self.RED + self.fmt + self.RESET,
            logging.CRITICAL: self.BOLD_RED + self.fmt + self.RESET,
        }
        self.datefmt = datefmt
        self.colors = colors

    def format(self, record):
        if self.colors:
            log_fmt = self.FORMATS.get(record.levelno)
        else:
            log_fmt = self.fmt
        if log_fmt is not None and (hasattr(record, "epoch") or hasattr(record, "update")):
            additional_formats = []
            if hasattr(record, "epoch") and hasattr(record, "max_epoch"):
                additional_formats.append(f"E={short_number_str(record.epoch)}/{short_number_str(record.max_epoch)}")
            if hasattr(record, "update") and hasattr(record, "max_update"):
                additional_formats.append(f"U={short_number_str(record.update)}/{short_number_str(record.max_update)}")
            if additional_formats:
                log_fmt = log_fmt.replace("%(message)s", f"{' '.join(additional_formats)} %(message)s")
        formatter = logging.Formatter(log_fmt, self.datefmt)
        return formatter.format(record)


def short_number_str(number: float, precision: int = 1) -> str:
    """Convert a number to a short string with SI prefix, using rounding.

    Example: 1234567 -> 1.2M
    Example: 123 -> 123
    Example: 1234 -> 1.2K
    Example: 0.1234 -> 123.4m
    Example: 0.000123 -> 123.0µ

    Args:
        number: The number to convert.
        precision: The number of decimal places to include.

    Returns:
        The short string representation of the number.
    """
    if type(number) is int and abs(number) < 1000:
        return str(number)

    if number == 0:
        return f"{0.0:.{precision}f}"

    sign = ""
    if number < 0:
        number = -number
        sign = "-"

    exponent = math.log10(number)

    # Get the "magnitude" (K=1, M=2, G=3... m=-1, µ=-2...)
    magnitude = int(math.floor(exponent / 3))

    # Scale the number (e.g., 999999 -> 999.999, magnitude 1 ('K'))
    scaled_number = number / (1000**magnitude) if magnitude != 0 else number

    # --- Rounding and Rollover Logic ---

    # Format the scaled number *first* to apply rounding
    format_str = f"{{:.{precision}f}}"
    rounded_str = format_str.format(scaled_number)
    rounded_number = float(rounded_str)

    # Check if rounding caused a rollover (e.g., 999.999 -> 1000.0)
    # We use a threshold slightly below 1000 to handle float imprecision
    if rounded_number >= 999.999999 and magnitude != 0:
        rounded_number /= 1000.0
        magnitude += 1
        # Re-format after rollover
        rounded_str = format_str.format(rounded_number)

    # --- Suffix Assignment ---
    if magnitude == 0:
        suffix = ""
    elif magnitude > 0:
        # Handle magnitude rollover (e.g., 'K' -> 'M')
        magnitude = min(magnitude, len(_SI_SUFFIXES_LARGE) - 1)
        suffix = _SI_SUFFIXES_LARGE[magnitude]
    else:  # magnitude < 0
        abs_magnitude = abs(magnitude)
        abs_magnitude = min(abs_magnitude, len(_SI_SUFFIXES_SMALL))
        suffix = _SI_SUFFIXES_SMALL[abs_magnitude - 1]

    return f"{sign}{rounded_str}{suffix}"


def summarize_indices_list(indices: list[int]) -> list[str]:
    """Summarize a list of indices into ranges.

    Example: [0, 1, 2, 3, 6, 7, 8] -> ["0-3", "6-8"]

    Args:
        indices: The list of indices to summarize.

    Returns:
        A list of strings representing the summarized indices.
    """
    if indices is None:
        return ["all"]
    if not indices:
        return []

        # Ensure indices are sorted and unique for groupby logic
    indices = sorted(set(indices))

    result = []
    # Group by the difference between the value and its index
    for k, g in groupby(enumerate(indices), key=lambda x: x[1] - x[0]):
        # g is an iterator of (index, value) tuples
        group = list(g)
        first_val = group[0][1]  # Get value from first item in group
        last_val = group[-1][1]  # Get value from last item in group

        if first_val == last_val:
            result.append(str(first_val))
        else:
            result.append(f"{first_val}-{last_val}")

    return result


def tensor_like_to_string(tensor_or_list: torch.Tensor | list[Any]) -> str:
    """Convert a list or a tensor to a string representation.

    Example: [1, 2, 3] -> "1, 2, 3"
    Example: [1.1, 2.2, 3.3] -> "1.10, 2.20, 3.30"

    Args:
        tensor: Tensor or list to be converted to a string.

    Returns:
        The string representation of the tensor or list.
    """
    if isinstance(tensor_or_list, torch.Tensor):
        tensor_data = tensor_or_list.cpu().numpy()
    elif isinstance(tensor_or_list, list):
        tensor_data = np.array(tensor_or_list)
    else:
        # Handle cases where it's neither, just in case:
        tensor_data = np.array(tensor_or_list)

    return np.array2string(tensor_data, precision=2, separator=", ", floatmode="fixed")


def dict_to_string(obj: dict[str, Any], item_seperator: str = "-") -> str:
    """Convert a dictionary to a string representation.

    Example: {epoch: 5, batch_size: 64} -> epoch=5-batchsize=64

    Args:
        obj: The dictionary to convert.
        item_seperator: The separator to use between items.

    Returns:
        The string representation of the dictionary.
    """
    return item_seperator.join(f"{k}={v}" for k, v in obj.items())


def float_to_scientific_notation(value: float, max_precision: int, remove_plus: bool = True) -> str:
    """Convert a float to scientific notation with a specified precision.

    Example: value = 0.0000032, max_precision = 10 -> "3.2e-6"

    Args:
        value: The float to convert.
        max_precision: The maximum number of decimal places to include.
        remove_plus: Whether to remove the '+' sign from the exponent.

    Returns:
        The scientific notation string representation of the float.
    """
    if value == 0.0:
        # Use f-string to respect precision, e.g., "0.00e0" for prec=2
        format_str = f"{{:.{max_precision}e}}"
        float_str = format_str.format(value)
    else:
        # Use 'g' format specifier
        format_string = f"{{:.{max_precision}g}}"
        float_str = format_string.format(value)

        if "e" not in float_str:
            # It was a number that 'g' didn't convert, force 'e'
            format_string = f"{{:.{max_precision}e}}"
            float_str = format_string.format(value)

    if remove_plus:
        float_str = float_str.replace("e+", "e")

    return float_str


def seconds_to_duration_str(total_seconds: float | int) -> str:
    """Convert a number of seconds to a duration string.

    Example: 60 * 60 * 24 * 14 + 1234 -> "14-10:20:34.00"

    Args:
        total_seconds: The number of seconds to convert.

    Returns:
        The string representation of the duration.
    """
    tenth_milliseconds = int((total_seconds - int(total_seconds)) * 100)
    total_seconds = int(total_seconds)
    seconds = total_seconds % 60
    minutes = total_seconds % 3600 // 60
    hours = total_seconds % 86400 // 3600
    days = total_seconds // 86400
    if days > 0:
        return f"{days}-{hours:02}:{minutes:02}:{seconds:02}.{tenth_milliseconds:02}"
    return f"{hours:02}:{minutes:02}:{seconds:02}.{tenth_milliseconds:02}"

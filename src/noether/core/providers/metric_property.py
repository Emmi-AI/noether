#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import fnmatch
import logging
from enum import Enum, auto


class Ordinality(Enum):
    """Defines the direction of improvement for a metric."""

    HIGHER_IS_BETTER = auto()
    LOWER_IS_BETTER = auto()
    NEUTRAL = auto()


class MetricPropertyProvider:
    """Provider for properties about metrics, mainly the ordinality of a metric, i.e., is it better if it is higher
    (e.g, accuracy of a classifier) or better if it is lower (e.g., a mean-squared error or any other loss).
    Additionally, a concept of "neutral" metrics is introduced, which are things that are logged throughout training
    but there is no concept of "best" metric (e.g., a learning rate that is scheduled, weight decay, ...).

    By default, the following patterns are handled:
    - "*loss*" (lower is better)
    - "*error*" (lower is better)
    - "*accuracy*" (higher is better)

    Callbacks or trainers can freely add patterns if they introduce new metrics that are not handled.

    The MetricPropertyProvider is used by the experiment tracker to automatically create summary values and log them
    to the online interface/log files and also by early stopping mechanisms to enable flexible early stopping where
    the early stopping criteria can use loss values, but also, e.g., validation accuracies.
    """

    _PATTERNS: list[tuple[str, Ordinality]] = list()

    @staticmethod
    def register_pattern(pattern: str, ordinality: Ordinality):
        """Allows users (Trainers/Callbacks) to add new patterns easily.

        Registering the same ``(pattern, ordinality)`` pair more than once is a no-op, so a
        callback that registers its metric namespace on construction can be instantiated
        repeatedly (e.g. across stages) without accumulating duplicate entries.
        """
        entry = (pattern, ordinality)
        if entry not in MetricPropertyProvider._PATTERNS:
            MetricPropertyProvider._PATTERNS.append(entry)

    @staticmethod
    def _register_defaults():
        """Registers the default known patterns."""
        # Neutral patterns typically come first as they are often namespace-based.
        neutral_patterns = [
            "profiler/*",
            "optim/*",
            "profiling/*",
            "ctx/*",
            "loss_weight/*",
            "gradient/*",
            "detach/*",
            "lossdiff/*",
            "train_time/*",
        ]
        for pattern in neutral_patterns:
            MetricPropertyProvider.register_pattern(pattern, Ordinality.NEUTRAL)

        # Ordinal patterns:
        MetricPropertyProvider.register_pattern("*accuracy*", Ordinality.HIGHER_IS_BETTER)
        MetricPropertyProvider.register_pattern("*loss*", Ordinality.LOWER_IS_BETTER)
        MetricPropertyProvider.register_pattern("*error*", Ordinality.LOWER_IS_BETTER)

    def __init__(self):
        self.logger = logging.getLogger(type(self).__name__)
        # Ensure defaults are registered only once, primarily for testing purposes:
        if not MetricPropertyProvider._PATTERNS:
            MetricPropertyProvider._register_defaults()

    @staticmethod
    def get_ordinality(key: str) -> Ordinality | None:
        """
        Determines the ordinality of a metric key based on registered patterns.
        The search is case-insensitive.
        """
        key_lower = key.lower()
        # Iterate through patterns until the first match is found
        for pattern, ordinality in MetricPropertyProvider._PATTERNS:
            if fnmatch.fnmatch(key_lower, pattern):
                return ordinality
        return None

    def is_neutral_key(self, key: str) -> bool:
        """Returns True if the passed key is a neutral metric."""
        return self.get_ordinality(key) == Ordinality.NEUTRAL

    def higher_is_better(self, key: str) -> bool:
        """
        Returns whether or not the passed key is better if the metric is higher.
        Logs a warning if the key is unmatched.

        Args:
            key: The key to check.
        """
        ordinality = self.get_ordinality(key)

        if ordinality is None:
            self.logger.warning(
                f"Metric '{key}' had no defined pattern in MetricPropertyProvider. Defaulting to higher_is_better=True."
            )
            return True

        return ordinality == Ordinality.HIGHER_IS_BETTER

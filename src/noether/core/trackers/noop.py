#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.trackers import BaseTracker


class NoopTracker(BaseTracker):
    """Dummy tracker that does nothing."""

    def _init(self, config: dict, output_uri: str, run_id: str):
        self.logger.debug("Not using any experiment tracker")

    def _log(self, data: dict):
        pass

    def _set_summary(self, key, value):
        pass

    def _update_summary(self, data: dict):
        pass

    def _close(self):
        pass

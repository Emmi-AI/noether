#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
from noether.core.trackers import TensorboardTrackerSchema, TrackioTrackerSchema, WandBTrackerSchema

AnyTracker = WandBTrackerSchema | TrackioTrackerSchema | TensorboardTrackerSchema

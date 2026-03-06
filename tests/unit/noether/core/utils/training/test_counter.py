#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest

from noether.core.utils.training.counter import UpdateCounter
from noether.core.utils.training.training_iteration import TrainingIteration


@pytest.fixture
def start_iter():
    """Standard start at 0."""
    return TrainingIteration(epoch=0, update=0, sample=0)


def test_init_success(start_iter):
    """Test standard initialization."""
    # End after 10 epochs (1000 updates)
    end_iter = TrainingIteration(epoch=10)

    counter = UpdateCounter(
        start_iteration=start_iter,
        end_iteration=end_iter,
        updates_per_epoch=100,
        effective_batch_size=32,
    )

    assert counter.updates_per_epoch == 100
    assert counter.effective_batch_size == 32
    # Verify end iteration was fully calculated (10 epochs = 1000 updates = 32000 samples)
    assert counter.end_iteration.update == 1000
    assert counter.end_iteration.sample == 32000
    assert not counter.is_finished


def test_init_raises_start_not_fully_specified():
    """Start iteration must be E, U, and S."""
    partial_start = TrainingIteration(epoch=0)  # Missing update/sample
    end_iter = TrainingIteration(epoch=10)

    with pytest.raises(ValueError, match="start_iteration must be fully specified"):
        UpdateCounter(partial_start, end_iter, 100, 32)


def test_init_raises_inconsistent_start():
    """
    Test the assertion:
    assert self.start_iteration == TrainingIteration(...).to_fully_specified(...)

    If we say we are at Epoch 1, but Update 0 (and updates_per_epoch is 100),
    this is mathematically impossible/inconsistent.
    """
    # Inconsistent: Epoch 1 should imply Update 100, but we say Update 0
    inconsistent_start = TrainingIteration(epoch=1, update=0, sample=0)
    end_iter = TrainingIteration(epoch=10)

    with pytest.raises(AssertionError):
        UpdateCounter(inconsistent_start, end_iter, 100, 32)


def test_init_raises_end_not_minimally_specified(start_iter):
    """End iteration must have at least one field set."""
    empty_end = TrainingIteration()  # All None

    with pytest.raises(ValueError, match="end_iteration must be minimally specified"):
        UpdateCounter(start_iter, empty_end, 100, 32)


def test_resume_training_init():
    """Test initializing from a resumed state (non-zero start)."""
    # Start at Epoch 5, Update 500, Sample 16000
    start = TrainingIteration(epoch=5, update=500, sample=16000)
    # Train for 5 more epochs (Total 10)
    # We specify end relative to 0 usually, but here end is absolute target
    end = TrainingIteration(epoch=10)

    counter = UpdateCounter(start, end, updates_per_epoch=100, effective_batch_size=32)

    assert counter.cur_iteration == start
    # Target should be fully populated
    assert counter.end_iteration.update == 1000


def test_properties_calculation(start_iter):
    """Test derived properties like is_full_epoch."""
    end_iter = TrainingIteration(epoch=1)
    # 10 updates per epoch
    counter = UpdateCounter(start_iter, end_iter, updates_per_epoch=10, effective_batch_size=1)

    # Start (0) is full epoch:
    assert counter.is_full_epoch is True
    assert counter.epoch_as_float == 0.0

    # Advance to middle:
    counter.next_update()  # Update 1
    assert counter.update == 1
    assert counter.is_full_epoch is False
    assert counter.epoch_as_float == 0.1

    # Advance to end of epoch:
    for _ in range(9):
        counter.next_update()

    assert counter.update == 10
    assert counter.is_full_epoch is True
    assert counter.epoch_as_float == 1.0


def test_increments(start_iter):
    """Test next_epoch, next_update, add_samples."""
    end_iter = TrainingIteration(epoch=5)
    counter = UpdateCounter(start_iter, end_iter, updates_per_epoch=10, effective_batch_size=2)

    assert counter.epoch == 0
    assert counter.update == 0
    assert counter.sample == 0

    counter.next_update()
    assert counter.update == 1

    counter.next_epoch()
    assert counter.epoch == 1

    counter.add_samples(32)
    assert counter.sample == 32


def test_is_finished_exact(start_iter):
    """Test finishing exactly on target."""
    # Target: 10 updates
    end_iter = TrainingIteration(update=10)

    # We set updates_per_epoch=100 (larger than target 10). This ensures the target Epoch remains 0.
    # If we used 10, target Epoch would be 1, requiring us to call next_epoch() too.
    counter = UpdateCounter(start_iter, end_iter, updates_per_epoch=100, effective_batch_size=1)
    assert not counter.is_finished

    # Move to 9:
    for _ in range(9):
        counter.add_samples(1)  # Important: Must increment samples too
        counter.next_update()
    assert not counter.is_finished

    # Move to 10:
    counter.add_samples(1)
    counter.next_update()

    # Now Update=10, Sample=10, Epoch=0. Target is Update=10, Sample=10, Epoch=0.
    # Should be finished.
    assert counter.is_finished


def test_is_finished_overshoot(start_iter):
    """Test finishing if we somehow go past target (robustness)."""
    end_iter = TrainingIteration(update=10)
    # Target: E0, U10, S10
    counter = UpdateCounter(start_iter, end_iter, updates_per_epoch=100, effective_batch_size=1)

    # Manually set to overshoot
    counter.cur_iteration.update = 11
    counter.cur_iteration.sample = 11  # Must also overshoot samples
    # Epoch 0 is fine (0 >= 0)

    assert counter.is_finished


def test_mid_epoch_resume_end_epoch():
    """Test that end_iteration.epoch is correct for mid-epoch resume.

    When resuming mid-epoch with an update-based end, the epoch should be derived from the absolute update count,
    not the delta. E.g. start=U71, end=U750, updates_per_epoch=150 -> end epoch = 5 (not 4).
    """
    start = TrainingIteration(epoch=0, update=71, sample=71)
    end = TrainingIteration(update=750)
    counter = UpdateCounter(start, end, updates_per_epoch=150, effective_batch_size=1)

    assert counter.end_iteration.epoch == 5
    assert counter.end_iteration.update == 750
    assert counter.end_iteration.sample == 750

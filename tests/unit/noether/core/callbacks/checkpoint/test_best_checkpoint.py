#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pytest

from noether.core.callbacks.checkpoint.best_checkpoint import BestCheckpointCallback

_MODULE_PATH = "noether.core.callbacks.checkpoint.best_checkpoint"


@pytest.fixture
def callback_deps():
    return {
        "trainer": Mock(),
        "model": Mock(),
        "data_container": Mock(),
        "tracker": Mock(),
        "log_writer": Mock(log_cache={}),
        "checkpoint_writer": Mock(save=Mock()),
        "metric_property_provider": Mock(),
    }


@pytest.fixture
def base_config():
    """Base configuration for BestCheckpointCallback."""
    return {
        "every_n_epochs": None,
        "every_n_updates": 1,
        "every_n_samples": None,
        "batch_size": None,
        "metric_key": "val/acc",
        "model_names": [],
        "model_name": None,
        "tolerances": None,
        "save_frozen_weights": False,
        "eval_callbacks": None,
    }


class TestBestCheckpointCallback:
    @pytest.mark.parametrize(
        ("higher_is_better", "current", "new", "expected"),
        [
            (True, 0.5, 0.6, True),
            (True, 0.5, 0.4, False),
            (False, 0.5, 0.4, True),
            (False, 0.5, 0.6, False),
            (True, -float("inf"), 0.1, True),
            (False, float("inf"), 0.1, True),
        ],
    )
    def test_is_new_best_model(self, callback_deps, base_config, higher_is_better, current, new, expected):
        callback_deps["metric_property_provider"].higher_is_better.return_value = higher_is_better
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        cb.best_metric_value = current
        assert cb._is_new_best_model(new) == expected

    def test_saves_best_and_tolerance_checkpoints(self, callback_deps, base_config):
        callback_deps["metric_property_provider"].higher_is_better.return_value = True
        base_config["tolerances"] = [1, 3]
        callback_deps["log_writer"].log_cache = {"val/acc": 0.9}

        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        cb.periodic_callback(interval_type="epoch")

        assert callback_deps["checkpoint_writer"].save.call_count == 1
        checkpoint = [call.kwargs["checkpoint_tag"] for call in callback_deps["checkpoint_writer"].save.call_args_list]
        assert "best_model.val.acc" in checkpoint

    def test_no_save_during_eval(self, callback_deps, base_config):
        """During post-training eval, the callback must not save a checkpoint —
        it would otherwise re-save the loaded model under a "best" filename
        because the in-memory `best_metric_value` starts at +/- inf in a fresh
        eval process (callback state is not restored from the source run)."""
        callback_deps["metric_property_provider"].higher_is_better.return_value = True
        # Even with a metric value that would be a "new best", the eval guard
        # must short-circuit before the metric is read.
        callback_deps["log_writer"].log_cache = {"val/acc": 0.9}

        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        cb.periodic_callback(interval_type="eval")

        callback_deps["checkpoint_writer"].save.assert_not_called()

    def test_tolerance_counter_increments_and_triggers_save(self, callback_deps, base_config):
        """
        Tests that tolerance behaves like a 'patience' counter.
        Config: Tolerance = 2.
        Expectation:
          - Fail 1 (Counter 1): OK (1 <= 2)
          - Fail 2 (Counter 2): OK (2 <= 2)
          - Fail 3 (Counter 3): Exceeded (3 > 2)
        """
        callback_deps["metric_property_provider"].higher_is_better.return_value = True
        base_config["tolerances"] = [2]

        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        cb.best_metric_value = 0.9

        # 1. First Failure:
        callback_deps["log_writer"].log_cache = {"val/acc": 0.85}
        cb.periodic_callback(interval_type="epoch")
        assert cb.tolerance_counter == 1
        assert cb.tolerances_is_exceeded.get(2, False) is False

        # 2. Second Failure (Tolerance limit reached, but not exceeded):
        cb.periodic_callback(interval_type="epoch")
        assert cb.tolerance_counter == 2
        assert cb.tolerances_is_exceeded.get(2, False) is False

        # 3. Third Failure (Exceeded):
        cb.periodic_callback(interval_type="epoch")
        assert cb.tolerance_counter == 3
        assert cb.tolerances_is_exceeded[2] is True
        assert cb.metric_at_exceeded_tolerance[2] == 0.85

        assert callback_deps["checkpoint_writer"].save.call_count == 1  # Only the tolerance checkpoint should be saved
        assert any(
            "best_model.val.acc.tolerance=2" in call.kwargs["checkpoint_tag"]
            for call in callback_deps["checkpoint_writer"].save.call_args_list
        )

    def test_tolerance_counter_resets_on_new_best(self, callback_deps, base_config):
        """Tolerance counter AND exceeded flags must reset on new best model."""
        callback_deps["metric_property_provider"].higher_is_better.return_value = True
        base_config["tolerances"] = [5]

        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)

        callback_deps["log_writer"].log_cache = {"val/acc": 0.9}
        cb.periodic_callback(interval_type="epoch")

        # Fail a few times:
        callback_deps["log_writer"].log_cache = {"val/acc": 0.85}
        cb.periodic_callback(interval_type="epoch")
        cb.periodic_callback(interval_type="epoch")
        assert cb.tolerance_counter == 2

        # New best:
        callback_deps["log_writer"].log_cache = {"val/acc": 0.95}
        cb.periodic_callback(interval_type="epoch")

        assert cb.tolerance_counter == 0
        # Important: Ensure the state dict for exceeded is also reset:
        assert all(v is False for v in cb.tolerances_is_exceeded.values())

    def test_state_dict_round_trip(self, callback_deps, base_config):
        callback_deps["metric_property_provider"].higher_is_better.return_value = True
        base_config["tolerances"] = [1]
        callback_deps["log_writer"].log_cache = {"val/acc": 0.9}

        cb1 = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        cb1.periodic_callback(interval_type="epoch")

        callback_deps["log_writer"].log_cache = {"val/acc": 0.85}
        cb1.periodic_callback(interval_type="epoch")  # Counter 1
        cb1.periodic_callback(interval_type="epoch")  # Counter 2 (Exceeded for tolerance 1)

        state = cb1.state_dict()
        cb2 = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        cb2.load_state_dict(state)

        assert cb2.best_metric_value == 0.9
        assert cb2.tolerance_counter == 2
        assert cb2.tolerances_is_exceeded == {1: True}

    def test_raises_on_missing_log_cache(self, callback_deps, base_config):
        callback_deps["log_writer"].log_cache = None
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        with pytest.raises(KeyError, match="Log cache is empty"):
            cb.periodic_callback(interval_type="epoch")

    def test_raises_on_missing_metric_key(self, callback_deps, base_config):
        callback_deps["log_writer"].log_cache = {"other": 0.5}
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        with pytest.raises(KeyError, match="couldn't find metric_key"):
            cb.periodic_callback(interval_type="epoch")

    def test_non_iterator_child_fires_on_new_best(self, monkeypatch, callback_deps, base_config):
        """A non-iterator child dispatches via ``periodic_callback`` only when a new best is detected.

        Verifies that:
          - each child's log_writer is a PrefixedLogWriter scoped to ``best=<metric_key>/``
          - ``before_training`` / ``after_training`` are forwarded unconditionally
          - ``periodic_callback`` is invoked only on the new-best step, not on non-improving steps or
            tolerance-exceeded saves
          - the child sees the parent's actual ``interval_type``
        """
        callback_deps["metric_property_provider"].higher_is_better.return_value = True
        base_config["tolerances"] = [1]

        child = Mock(spec=["before_training", "periodic_callback", "after_training"])

        captured_kwargs: dict = {}

        def fake_create_list(_configs, **child_kwargs):
            captured_kwargs.update(child_kwargs)
            return [child]

        fake_factory = Mock()
        fake_factory.create_list.side_effect = fake_create_list
        monkeypatch.setattr(_MODULE_PATH + ".Factory", Mock(return_value=fake_factory))

        base_config["eval_callbacks"] = [object()]  # sentinel; Factory is mocked

        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)

        from noether.core.writers import PrefixedLogWriter

        assert isinstance(captured_kwargs["log_writer"], PrefixedLogWriter)
        assert captured_kwargs["log_writer"]._prefix == "best=val.acc"

        update_counter = SimpleNamespace(cur_iteration=SimpleNamespace(sample=0))
        cb.before_training(update_counter=update_counter, extra="forwarded")
        child.before_training.assert_called_once()
        assert child.before_training.call_args.kwargs["extra"] == "forwarded"

        # New best -> child.periodic_callback fires with the parent's actual interval_type.
        callback_deps["log_writer"].log_cache = {"val/acc": 0.9}
        cb.periodic_callback(interval_type="epoch", update_counter=update_counter, trainer_model="m")
        assert child.periodic_callback.call_count == 1
        assert child.periodic_callback.call_args.kwargs["update_counter"] is update_counter
        assert child.periodic_callback.call_args.kwargs["interval_type"] == "epoch"

        # Non-improving step -> no extra child dispatch.
        callback_deps["log_writer"].log_cache = {"val/acc": 0.85}
        cb.periodic_callback(interval_type="epoch", update_counter=update_counter)
        assert child.periodic_callback.call_count == 1

        # Tolerance-exceeded save -> still no child dispatch (only main best triggers children).
        cb.periodic_callback(interval_type="epoch", update_counter=update_counter)
        assert cb.tolerances_is_exceeded[1] is True
        assert child.periodic_callback.call_count == 1

        cb.after_training(extra="forwarded")
        child.after_training.assert_called_once()
        assert child.after_training.call_args.kwargs["extra"] == "forwarded"

    def test_iterator_child_gets_fresh_loader_on_new_best(self, monkeypatch, callback_deps, base_config):
        """Iterator children receive their own fresh ``data_iter`` from a standalone ``DataLoader``
        built off their ``sampler_config`` — they do NOT consume from the trainer's shared
        ``data_iter`` (which is why we don't expose them via ``get_children``).
        """
        from noether.core.callbacks.periodic import PeriodicDataIteratorCallback

        callback_deps["metric_property_provider"].higher_is_better.return_value = True

        # Non-distributed sampler path: sampler has ``data_source`` and isinstance(_, DistributedSampler)
        # is False, so ``_build_loader_for_child`` reads the dataset from ``data_source``.
        fake_dataset = object()
        fake_sampler = SimpleNamespace(data_source=fake_dataset)

        child = Mock(spec=PeriodicDataIteratorCallback)
        child.sampler_config = SimpleNamespace(sampler=fake_sampler, batch_size=None, pipeline="pipeline_sentinel")

        # Capture the DataLoader kwargs and stub iteration so the assertion can check the iterator
        # handed to ``at_eval`` came from this loader.
        captured_loader_kwargs: dict = {}

        class FakeLoader:
            def __iter__(self):
                return iter([("batch0",), ("batch1",)])

        def fake_DataLoader(**kwargs):
            captured_loader_kwargs.update(kwargs)
            return FakeLoader()

        monkeypatch.setattr(_MODULE_PATH + ".DataLoader", fake_DataLoader)

        fake_factory = Mock()
        fake_factory.create_list.return_value = [child]
        monkeypatch.setattr(_MODULE_PATH + ".Factory", Mock(return_value=fake_factory))

        base_config["eval_callbacks"] = [object()]
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)

        update_counter = SimpleNamespace(cur_iteration=SimpleNamespace(sample=0))
        callback_deps["log_writer"].log_cache = {"val/acc": 0.9}
        # The trainer always passes ``batch_size`` and ``trainer_model`` to every PeriodicCallback hook;
        # ``data_iter`` is *not* passed to us (we hide iterator children from ``get_children``, so
        # ``_needs_iterator_args`` returns False). We build the child's ``data_iter`` ourselves.
        cb.periodic_callback(
            interval_type="update",
            update_counter=update_counter,
            trainer_model="dist_model",
            batch_size=8,
        )

        # DataLoader built from the child's sampler_config — sampler, pipeline, batch_size pass through.
        assert captured_loader_kwargs["dataset"] is fake_dataset
        assert captured_loader_kwargs["sampler"] is fake_sampler
        assert captured_loader_kwargs["batch_size"] == 8
        assert captured_loader_kwargs["collate_fn"] == "pipeline_sentinel"

        # Child receives the fresh iter() and the forwarded trainer args.
        assert child.periodic_callback.call_count == 1
        kwargs = child.periodic_callback.call_args.kwargs
        assert kwargs["interval_type"] == "update"  # parent's interval_type, not "eval"
        assert kwargs["update_counter"] is update_counter
        assert kwargs["trainer_model"] == "dist_model"
        assert kwargs["batch_size"] == 8
        assert next(kwargs["data_iter"]) == ("batch0",)

    def test_children_fire_on_eval_interval(self, monkeypatch, callback_deps, base_config):
        """When the trainer calls ``at_eval`` (post-training eval), children dispatch unconditionally
        against the loaded best model — no new-best detection, no checkpoint save, no tolerance update.
        """
        from noether.core.callbacks.periodic import PeriodicDataIteratorCallback

        callback_deps["metric_property_provider"].higher_is_better.return_value = True

        iterator_child = Mock(spec=PeriodicDataIteratorCallback)
        iterator_child.sampler_config = SimpleNamespace(
            sampler=SimpleNamespace(data_source=object()), batch_size=2, pipeline=None
        )
        non_iterator_child = Mock(spec=["periodic_callback"])

        class FakeLoader:
            def __iter__(self):
                return iter([])

        monkeypatch.setattr(_MODULE_PATH + ".DataLoader", lambda **_: FakeLoader())

        fake_factory = Mock()
        fake_factory.create_list.return_value = [iterator_child, non_iterator_child]
        monkeypatch.setattr(_MODULE_PATH + ".Factory", Mock(return_value=fake_factory))

        base_config["eval_callbacks"] = [object(), object()]
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)

        update_counter = SimpleNamespace()
        cb.periodic_callback(
            interval_type="eval",
            update_counter=update_counter,
            trainer_model="dist_model",
            batch_size=2,
        )

        # Both children invoked with interval_type="eval"; no checkpoint save.
        assert iterator_child.periodic_callback.call_count == 1
        assert iterator_child.periodic_callback.call_args.kwargs["interval_type"] == "eval"
        assert non_iterator_child.periodic_callback.call_count == 1
        assert non_iterator_child.periodic_callback.call_args.kwargs["interval_type"] == "eval"
        callback_deps["checkpoint_writer"].save.assert_not_called()

    def test_iterator_child_loader_is_cached(self, monkeypatch, callback_deps, base_config):
        """The DataLoader is built once per child and reused across new-best events."""
        from noether.core.callbacks.periodic import PeriodicDataIteratorCallback

        callback_deps["metric_property_provider"].higher_is_better.return_value = True

        fake_sampler = SimpleNamespace(data_source=object())
        child = Mock(spec=PeriodicDataIteratorCallback)
        child.sampler_config = SimpleNamespace(sampler=fake_sampler, batch_size=4, pipeline=None)

        construction_count = 0

        class FakeLoader:
            def __iter__(self):
                return iter([])

        def fake_DataLoader(**_kwargs):
            nonlocal construction_count
            construction_count += 1
            return FakeLoader()

        monkeypatch.setattr(_MODULE_PATH + ".DataLoader", fake_DataLoader)

        fake_factory = Mock()
        fake_factory.create_list.return_value = [child]
        monkeypatch.setattr(_MODULE_PATH + ".Factory", Mock(return_value=fake_factory))

        base_config["eval_callbacks"] = [object()]
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)

        # Two consecutive new-best events.
        callback_deps["log_writer"].log_cache = {"val/acc": 0.5}
        cb.periodic_callback(interval_type="update", trainer_model="m", batch_size=4)
        callback_deps["log_writer"].log_cache = {"val/acc": 0.9}
        cb.periodic_callback(interval_type="update", trainer_model="m", batch_size=4)

        assert construction_count == 1
        assert child.periodic_callback.call_count == 2

    def test_iterator_child_not_fired_on_non_best(self, monkeypatch, callback_deps, base_config):
        """No DataLoader construction and no child dispatch on a non-new-best step — there's no
        shared stream to keep aligned, so we simply do nothing for the child."""
        from noether.core.callbacks.periodic import PeriodicDataIteratorCallback

        callback_deps["metric_property_provider"].higher_is_better.return_value = True

        child = Mock(spec=PeriodicDataIteratorCallback)
        child.sampler_config = SimpleNamespace(
            sampler=SimpleNamespace(data_source=object()), batch_size=None, pipeline=None
        )

        constructed = False

        def fake_DataLoader(**_):
            nonlocal constructed
            constructed = True
            return SimpleNamespace(__iter__=lambda self: iter([]))

        monkeypatch.setattr(_MODULE_PATH + ".DataLoader", fake_DataLoader)

        fake_factory = Mock()
        fake_factory.create_list.return_value = [child]
        monkeypatch.setattr(_MODULE_PATH + ".Factory", Mock(return_value=fake_factory))

        base_config["eval_callbacks"] = [object()]
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        cb.best_metric_value = 0.95  # any new metric below this is non-improving

        callback_deps["log_writer"].log_cache = {"val/acc": 0.5}
        cb.periodic_callback(interval_type="update", trainer_model="m", batch_size=4)

        assert child.periodic_callback.call_count == 0
        assert constructed is False

    def test_get_children_excludes_iterator_children(self, monkeypatch, callback_deps, base_config):
        """``get_children`` hides iterator children (trainer must not register their samplers on the
        shared ``InterleavedSampler``) but still exposes non-iterator children."""
        from noether.core.callbacks.periodic import PeriodicDataIteratorCallback

        callback_deps["metric_property_provider"].higher_is_better.return_value = True

        iterator_child = Mock(spec=PeriodicDataIteratorCallback)
        iterator_child.sampler_config = SimpleNamespace(
            sampler=SimpleNamespace(data_source=object()), batch_size=1, pipeline=None
        )
        non_iterator_child = Mock()

        fake_factory = Mock()
        fake_factory.create_list.return_value = [iterator_child, non_iterator_child]
        monkeypatch.setattr(_MODULE_PATH + ".Factory", Mock(return_value=fake_factory))

        base_config["eval_callbacks"] = [object(), object()]
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)

        assert cb.get_children() == [non_iterator_child]

    def test_no_eval_callbacks_when_unconfigured(self, callback_deps, base_config):
        callback_deps["metric_property_provider"].higher_is_better.return_value = True
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)
        assert cb.eval_callbacks == []
        assert cb.get_children() == []

    def test_build_loader_for_distributed_sampler_reads_dataset_attr(self, monkeypatch, callback_deps, base_config):
        """Distributed sampler path of ``_build_loader_for_child``: when the child's sampler is a
        ``DistributedSampler``, the dataset is read from ``sampler.dataset`` (not ``data_source``).
        """
        from noether.core.callbacks.periodic import PeriodicDataIteratorCallback

        callback_deps["metric_property_provider"].higher_is_better.return_value = True

        # Patch the module's DistributedSampler to a class we control so isinstance(..., DistributedSampler)
        # resolves to True for our fake sampler without needing a torch.distributed setup.
        class FakeDistributedSampler:
            def __init__(self, dataset):
                self.dataset = dataset

        monkeypatch.setattr(_MODULE_PATH + ".DistributedSampler", FakeDistributedSampler)

        fake_dataset = object()
        fake_sampler = FakeDistributedSampler(dataset=fake_dataset)
        child = Mock(spec=PeriodicDataIteratorCallback)
        child.sampler_config = SimpleNamespace(sampler=fake_sampler, batch_size=2, pipeline=None)

        captured_loader_kwargs: dict = {}

        class FakeLoader:
            def __iter__(self):
                return iter([])

        def fake_DataLoader(**kwargs):
            captured_loader_kwargs.update(kwargs)
            return FakeLoader()

        monkeypatch.setattr(_MODULE_PATH + ".DataLoader", fake_DataLoader)

        fake_factory = Mock()
        fake_factory.create_list.return_value = [child]
        monkeypatch.setattr(_MODULE_PATH + ".Factory", Mock(return_value=fake_factory))

        base_config["eval_callbacks"] = [object()]
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)

        callback_deps["log_writer"].log_cache = {"val/acc": 0.9}
        cb.periodic_callback(interval_type="update", trainer_model="m", batch_size=4)

        assert captured_loader_kwargs["dataset"] is fake_dataset

    def test_before_after_training_reach_iterator_children(self, monkeypatch, callback_deps, base_config):
        """Both iterator and non-iterator children receive ``before_training`` / ``after_training``."""
        from noether.core.callbacks.periodic import PeriodicDataIteratorCallback

        callback_deps["metric_property_provider"].higher_is_better.return_value = True

        iter_child = Mock(spec=PeriodicDataIteratorCallback)
        iter_child.sampler_config = SimpleNamespace(
            sampler=SimpleNamespace(data_source=object()), batch_size=1, pipeline=None
        )
        non_iter_child = Mock(spec=["before_training", "periodic_callback", "after_training"])

        fake_factory = Mock()
        fake_factory.create_list.return_value = [iter_child, non_iter_child]
        monkeypatch.setattr(_MODULE_PATH + ".Factory", Mock(return_value=fake_factory))

        base_config["eval_callbacks"] = [object(), object()]
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)

        update_counter = SimpleNamespace(cur_iteration=SimpleNamespace(sample=0))
        cb.before_training(update_counter=update_counter)
        cb.after_training()

        iter_child.before_training.assert_called_once()
        iter_child.after_training.assert_called_once()
        non_iter_child.before_training.assert_called_once()
        non_iter_child.after_training.assert_called_once()

    def test_each_iterator_child_gets_distinct_cached_loader(self, monkeypatch, callback_deps, base_config):
        """With multiple iterator children, the cache builds one loader per child (keyed by ``id``)."""
        from noether.core.callbacks.periodic import PeriodicDataIteratorCallback

        callback_deps["metric_property_provider"].higher_is_better.return_value = True

        child_a = Mock(spec=PeriodicDataIteratorCallback)
        child_a.sampler_config = SimpleNamespace(
            sampler=SimpleNamespace(data_source=object()), batch_size=1, pipeline=None
        )
        child_b = Mock(spec=PeriodicDataIteratorCallback)
        child_b.sampler_config = SimpleNamespace(
            sampler=SimpleNamespace(data_source=object()), batch_size=1, pipeline=None
        )

        constructed_loaders: list[object] = []

        class FakeLoader:
            def __iter__(self):
                return iter([])

        def fake_DataLoader(**_):
            loader = FakeLoader()
            constructed_loaders.append(loader)
            return loader

        monkeypatch.setattr(_MODULE_PATH + ".DataLoader", fake_DataLoader)

        fake_factory = Mock()
        fake_factory.create_list.return_value = [child_a, child_b]
        monkeypatch.setattr(_MODULE_PATH + ".Factory", Mock(return_value=fake_factory))

        base_config["eval_callbacks"] = [object(), object()]
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)

        # Two consecutive new-best events -> 2 loaders total (one per child), each reused.
        callback_deps["log_writer"].log_cache = {"val/acc": 0.5}
        cb.periodic_callback(interval_type="update", trainer_model="m", batch_size=4)
        callback_deps["log_writer"].log_cache = {"val/acc": 0.9}
        cb.periodic_callback(interval_type="update", trainer_model="m", batch_size=4)

        assert len(constructed_loaders) == 2
        assert constructed_loaders[0] is not constructed_loaders[1]
        assert child_a.periodic_callback.call_count == 2
        assert child_b.periodic_callback.call_count == 2

    def test_mixed_children_dispatch_on_new_best(self, monkeypatch, callback_deps, base_config):
        """On a new-best step, an iterator child gets ``data_iter`` injected; a non-iterator child
        does not. Both still receive ``trainer_model`` / ``batch_size`` from the trainer's kwargs.
        """
        from noether.core.callbacks.periodic import PeriodicDataIteratorCallback

        callback_deps["metric_property_provider"].higher_is_better.return_value = True

        iter_child = Mock(spec=PeriodicDataIteratorCallback)
        iter_child.sampler_config = SimpleNamespace(
            sampler=SimpleNamespace(data_source=object()), batch_size=1, pipeline=None
        )
        non_iter_child = Mock(spec=["periodic_callback"])

        class FakeLoader:
            def __iter__(self):
                return iter([])

        monkeypatch.setattr(_MODULE_PATH + ".DataLoader", lambda **_: FakeLoader())

        fake_factory = Mock()
        fake_factory.create_list.return_value = [iter_child, non_iter_child]
        monkeypatch.setattr(_MODULE_PATH + ".Factory", Mock(return_value=fake_factory))

        base_config["eval_callbacks"] = [object(), object()]
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)

        callback_deps["log_writer"].log_cache = {"val/acc": 0.9}
        cb.periodic_callback(interval_type="update", trainer_model="dist_model", batch_size=4)

        iter_kwargs = iter_child.periodic_callback.call_args.kwargs
        assert "data_iter" in iter_kwargs
        assert iter_kwargs["trainer_model"] == "dist_model"
        assert iter_kwargs["batch_size"] == 4

        non_iter_kwargs = non_iter_child.periodic_callback.call_args.kwargs
        assert "data_iter" not in non_iter_kwargs
        assert non_iter_kwargs["trainer_model"] == "dist_model"
        assert non_iter_kwargs["batch_size"] == 4

    def test_eval_interval_does_not_require_log_cache(self, monkeypatch, callback_deps, base_config):
        """The ``"eval"`` interval short-circuits before reading ``log_cache``, so children still
        dispatch even if no metric has been logged in the current process."""
        callback_deps["metric_property_provider"].higher_is_better.return_value = True

        child = Mock(spec=["periodic_callback"])

        fake_factory = Mock()
        fake_factory.create_list.return_value = [child]
        monkeypatch.setattr(_MODULE_PATH + ".Factory", Mock(return_value=fake_factory))

        base_config["eval_callbacks"] = [object()]
        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)

        # Log cache is None (would raise KeyError on a non-eval interval).
        callback_deps["log_writer"].log_cache = None
        cb.periodic_callback(interval_type="eval", trainer_model="m", batch_size=4)

        assert child.periodic_callback.call_count == 1
        callback_deps["checkpoint_writer"].save.assert_not_called()

    def test_after_training_logs_tolerance_metrics(self, callback_deps, base_config):
        callback_deps["metric_property_provider"].higher_is_better.return_value = True
        base_config["tolerances"] = [1]

        cb = BestCheckpointCallback(callback_config=SimpleNamespace(**base_config), **callback_deps)

        with patch(
            "noether.core.callbacks.checkpoint.best_checkpoint.BestCheckpointCallback.logger", new_callable=PropertyMock
        ) as mock_logger:
            mock_log_instance = Mock()
            mock_logger.return_value = mock_log_instance

            callback_deps["log_writer"].log_cache = {"val/acc": 0.9}
            cb.periodic_callback(interval_type="epoch")

            callback_deps["log_writer"].log_cache = {"val/acc": 0.85}
            cb.periodic_callback(interval_type="epoch")  # Counter 1
            cb.periodic_callback(interval_type="epoch")  # Counter 2 (Exceeds 1)

            cb.after_training()

            log_calls = [call[0][0] for call in mock_log_instance.info.call_args_list]
            assert any("tolerance=1" in log for log in log_calls)

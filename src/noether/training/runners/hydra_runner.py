#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import os
import platform
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torch.distributed import broadcast_object_list

from noether.core.configs import Hyperparameters
from noether.core.distributed import (
    barrier,
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    is_managed,
    is_rank0,
    run_managed,
    run_unmanaged,
)
from noether.core.factory import DatasetFactory, Factory, class_constructor_from_class_path
from noether.core.models import ModelBase
from noether.core.providers import MetricPropertyProvider, PathProvider
from noether.core.schemas.initializers import PreviousRunInitializerConfig, ResumeInitializerConfig
from noether.core.schemas.schema import ConfigSchema
from noether.core.trackers import NoopTracker
from noether.core.trackers.base import BaseTracker
from noether.core.utils.code import store_code_archive
from noether.core.utils.logging import add_global_handlers, log_from_all_ranks
from noether.core.utils.logging.base import MessageCounter
from noether.core.utils.platform import get_cli_command, log_system_info
from noether.core.utils.seed import set_seed
from noether.data import Collator, Dataset, MultiStagePipeline
from noether.data.base.wrapper import DatasetWrapper
from noether.data.container import DataContainer
from noether.training.trainers import BaseTrainer

logger = logging.getLogger(__name__)


class HydraRunner:
    """Runs an experiment using @hydra.main as entry point."""

    def run(self, hydra_config: dict) -> None:
        """Runs an experiment. Given CLI args which contain a path to a hyperparameter file.

        Args:
            config: hydra config as a YAML.

        """

        # get config schema
        config_schema = class_constructor_from_class_path(hydra_config["config_schema_kind"])
        config: ConfigSchema = config_schema(**hydra_config)

        # initialize loggers for setup
        add_global_handlers(log_file_uri=None, debug=config.debug)
        logger.info(get_cli_command())

        # immediately log hostname to see which nodes are unhealthy if run crashes immediately
        logger.info(f"Started training on {platform.uname().node}")

        if is_managed():
            run_managed(
                accelerator=config.accelerator,
                devices=config.devices,
                main=partial(self.main, config=config),
            )
        else:
            # retrieve devices string (use "0" by default which will use the first GPU, i.e., cuda:0)
            run_unmanaged(
                accelerator=str(config.accelerator),
                devices=config.devices,
                main=partial(self.main, config=config),
                master_port=config.master_port,
            )

    @staticmethod
    def _extract_name_from_overrides(overrides: list[str]) -> Iterator[str]:
        for override in overrides:
            if any(override.startswith(key) for key in ("accelerator", "devices", "tracker", "name")):
                continue

            # 2. Apply transformations to the items we keep:
            clean_override = override.replace("+", "")

            # Shorten numbers (e.g., 1.0 -> 1, 0.1 -> 01):
            if "=" in clean_override:
                pre_eq, post_eq = clean_override.split("=", 1)
                post_eq = post_eq.removesuffix(".0")
                post_eq = post_eq.removesuffix(".")

                # Shorten common fields (e.g., clip_grad_norm -> gclip):
                if pre_eq == "clip_grad_norm":
                    pre_eq = "gclip"

                clean_override = f"{pre_eq}={post_eq}"

            yield clean_override

    @staticmethod
    def derive_run_name(name: str):
        """Derives a run name by appending hydra overrides to the given name."""
        try:
            overrides = OmegaConf.to_container(hydra.core.hydra_config.HydraConfig.get().overrides.task, resolve=True)  # type: ignore
            if not isinstance(overrides, list):
                raise TypeError("overrides is expected to be a list of strings")
            new_overrides = list(HydraRunner._extract_name_from_overrides(overrides))

            if new_overrides:
                logger.info(f"{len(new_overrides)} overrides set: {' '.join(new_overrides)}")
                return f"{name}--{'-'.join(new_overrides)}"

        except ValueError:
            logger.warning(
                "overrides could not be included into name -> this occours if starting multi-GPU runs without SLURM "
                "as the hydra config is not initialized in spawned processes"
            )
        return name

    @staticmethod
    def main(device: str, config: ConfigSchema) -> None:
        """Main method called from each GPU main process after being spawned and initialized for communication.

        Args:
            device: device of the current process.
            config: configuration of the experiment.
        """
        trainer, model, tracker, message_counter = HydraRunner.setup_experiment(device=device, config=config)

        trainer.train(model)
        tracker.summarize_logvalues()

        # cleanup
        message_counter.log()
        tracker.close()

    @staticmethod
    def setup_experiment(
        device: str,
        config: ConfigSchema,
        initializer_config_class: type[ResumeInitializerConfig]
        | type[PreviousRunInitializerConfig] = ResumeInitializerConfig,
    ) -> tuple[BaseTrainer, ModelBase, BaseTracker, MessageCounter]:
        """Sets up the experiment objects (datasets, trainer, model, etc.)."""
        world_size = get_world_size()
        if world_size > 1:
            with log_from_all_ranks():
                logger.debug(f"initialized process rank={get_rank()} local_rank={get_local_rank()} pid={os.getpid()}")
            barrier()
            logger.debug(f"Successfully initialized {get_world_size()} processes")

        # cudnn
        if config.accelerator == "gpu":
            if config.cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                assert not config.cudnn_deterministic, "cudnn_benchmark can make things non-deterministic"
            else:
                logger.warning("disabled cudnn benchmark")
                if config.cudnn_deterministic:
                    torch.backends.cudnn.deterministic = True
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                    logger.warning("enabled cudnn deterministic")

        # retrieve run_id from hp (allows queueing up dependent stages by referencing stage_ids in the yamls)
        if (run_id := config.run_id) is None:
            run_id = PathProvider.generate_run_id()
            if is_distributed():
                object_list = [run_id] if is_rank0() else [0]  # type: ignore
                broadcast_object_list(object_list, src=0)
                run_id = object_list[0]
                assert run_id is not None, "run_id should have been broadcasted to all ranks"

        # initialize path where to store logs/checkpoints/...
        output_path = config.output_path

        # initialize logging
        path_provider = PathProvider(
            output_root_path=output_path,
            run_id=run_id,
            stage_name=config.stage_name,
            debug=config.debug,
        )

        resume_run_id: str | None = config.resume_run_id
        if resume_run_id is not None:
            resume_checkpoint: str | None = config.resume_checkpoint
            if resume_checkpoint is None:
                checkpoint: str | dict = "latest"
            elif resume_checkpoint.startswith("E"):
                checkpoint = dict(epoch=int(resume_checkpoint[1:]))
            elif resume_checkpoint.startswith("U"):
                checkpoint = dict(update=int(resume_checkpoint[1:]))
            elif resume_checkpoint.startswith("S"):
                checkpoint = dict(sample=int(resume_checkpoint[1:]))
            else:
                # any checkpoint (like cp=last or cp=best.accuracy1.test.main)
                checkpoint = resume_checkpoint

            ancesetor = path_provider.with_run(
                run_id=resume_run_id,
                stage_name=config.resume_stage_name,
            )
            path_provider.link(ancesetor)

            config.trainer.initializer = initializer_config_class.model_validate(
                dict(
                    run_id=resume_run_id,
                    stage_name=config.resume_stage_name,
                    checkpoint_tag=checkpoint,
                    model_name=config.model.name,
                )
            )

        message_counter = add_global_handlers(log_file_uri=path_provider.logfile_uri, debug=config.debug)

        run_name = HydraRunner.derive_run_name(config.name or "")

        logger.info(f"Running experiment name: {run_name}, run_id: {run_id}, stage_name: {config.stage_name}")

        seed = config.seed
        tracker_config = config.tracker

        metric_property_provider = MetricPropertyProvider()

        # allow disabling via hydra override "+tracker=disabled"
        if tracker_config is None or config.debug:
            tracker: BaseTracker = NoopTracker(
                metric_property_provider=metric_property_provider,
                path_provider=path_provider,
            )
        else:
            t = Factory().create(
                tracker_config,
                metric_property_provider=metric_property_provider,
                path_provider=path_provider,
            )
            if not isinstance(t, BaseTracker):
                raise TypeError(f"tracker is expected to be of type BaseTracker but got {type(t)}")
            tracker = t

        tracker.init(
            accelerator=str(config.accelerator),
            run_name=run_name,
            stage_hp=config.model_dump(),
            stage_name=config.stage_name,
            run_id=run_id,
            output_uri=path_provider.run_output_path.as_posix(),
        )

        log_system_info()

        if is_rank0():
            path = path_provider.run_output_path / "hp_resolved.yaml"
            Hyperparameters.save_resolved(config, path)
            Hyperparameters.log(config)

        if is_distributed():
            # using a different seed for every rank to ensure that stochastic processes are different across ranks
            # for large batch_sizes this shouldn't matter too much
            # this is relevant for:
            # - augmentations (augmentation parameters of sample0 of rank0 == augparams of sample0 of rank1 == ...)
            # - the masks of a MAE are the same for every rank
            # NOTE: DDP syncs the parameters in its __init__ method -> same initial parameters independent of seed
            seed += get_rank()
        set_seed(seed)

        # init datasets
        datasets = {}

        for dataset_key, dataset_config in config.datasets.items():
            if dataset_key in datasets:
                raise KeyError(
                    f"Dataset '{dataset_key}' is already initialized, which means that the same dataset key is configured twice. Cannot use the same dataset key twice"
                )

            logger.info(f"Initializing dataset {dataset_key}")
            dataset: Dataset | None = DatasetFactory().create(dataset_config)

            if not isinstance(dataset, (Dataset, DatasetWrapper)):
                raise TypeError(f"dataset is expected to be of type Dataset but got {type(dataset)}")
            if dataset_config.pipeline is None:
                pipeline: Collator = MultiStagePipeline()
            else:
                p = Factory().create(dataset_config.pipeline)
                if not isinstance(p, Collator | MultiStagePipeline):
                    raise TypeError(f"pipeline is expected to be of type Collator but got {type(p)}")
                pipeline = p

            dataset.pipeline = pipeline
            datasets[dataset_key] = dataset

        data_container = DataContainer(datasets=datasets, num_workers=config.num_workers, pin_memory=device == "cuda")

        # init trainer
        trainer = Factory().create(
            config.trainer,
            data_container=data_container,
            device=device,
            tracker=tracker,
            path_provider=path_provider,
            metric_property_provider=metric_property_provider,
        )

        logger.info(f"Initialized trainer {type(trainer).__name__}")
        if not isinstance(trainer, BaseTrainer):
            raise TypeError(f"trainer is expected to be of type BaseTrainer but got {type(trainer)}")

        # init model
        if config.model is None:
            raise ValueError("No model defined in config")

        model = Factory().instantiate(
            config.model,
            data_container=data_container,
            update_counter=trainer.update_counter,
            path_provider=path_provider,
        )
        logger.info(f"Initialized model {type(model).__name__}")
        if not isinstance(model, ModelBase):
            raise TypeError(f"Model is expected to be of type ModelBase but got {type(model)}")

        logger.debug(f"Model definition:\n{model}")

        if config.store_code_in_output:
            if is_rank0():
                if archive_path := store_code_archive(code_path=Path.cwd(), output_path=path_provider.run_output_path):
                    logger.debug(f"Stored code in {archive_path.as_posix()}")
            if is_distributed():
                barrier()

        return trainer, model, tracker, message_counter

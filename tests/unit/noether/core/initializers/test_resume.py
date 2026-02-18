#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
from __future__ import annotations

from pathlib import Path

from noether.core.initializers.resume import ResumeInitializer
from noether.core.providers.path import PathProvider
from noether.core.schemas.initializers import ResumeInitializerConfig


def test_resume_initializer_init():
    run_id = "stage_1"
    model_name = "model_1"
    checkpoint_tag = "latest"
    model_info = "info_1"

    config = ResumeInitializerConfig(
        run_id="stage_1", model_name="model_1", checkpoint_tag="latest", model_info="info_1"
    )
    initializer = ResumeInitializer(initializer_config=config, path_provider=PathProvider(Path("."), "1"))
    assert initializer.run_id == run_id
    assert initializer.model_name == model_name
    assert initializer.checkpoint_tag == checkpoint_tag
    assert initializer.model_info == model_info
    assert initializer.load_optim is True

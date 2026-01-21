#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

InitWeightsMode = Literal["truncnormal002", "torch", "truncnormal", "truncnormal002-identity", "torchs", "zeros"]

ActivationTypes = Literal["GELU", "TANH", "SIGMOID", "RELU", "LEAKY_RELU", "SOFTPLUS", "ELU", "SILU"]


class CheckpointKeys:
    """
    Defines the standard, possible keys in the checkpoint dict.
    """

    STATE_DICT = "state_dict"
    """ The pytorch state dict of the model. I.e. the model weights/tensors/buffers. """
    CHECKPOINT_TAG = "checkpoint_tag"
    """ The checkpoint tag, e.g., "E10_U200_S800" or "latest". """
    TRAINING_ITERATION = "training_iteration"
    """ The detailed information about training iteration as a dict with keys 'epoch', 'update', and 'sample'. """
    RUN_ID = "run_id"
    """ The ID of the run from which this checkpoint was created. """
    MODEL_CONFIG = "model_config"
    """ The model configuration used to instantiate the model. A serialized dict of the pydantic model config. """
    CONFIG_KIND = "config_kind"
    """ The kind (i.e., class path) of the model configuration. Used to instantiate the correct model configuration. """
    CALLBACK_STATE_DICT = "callback_state_dicts"
    """ The state dicts of the callbacks. """
    GRAD_SCALER = "grad_scaler"
    """ The state dict of the grad scaler (if used). """

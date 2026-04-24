#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from abc import ABC
from collections import OrderedDict
from collections.abc import Sequence
from typing import Annotated, Any, ClassVar, Literal, TypeVar, Union

from pydantic import BaseModel, Field, RootModel, model_validator

from noether.core.schemas.lib import Discriminated, _RegistryBase
from noether.core.schemas.normalizers import NormalizerConfig


class DatasetWrapperConfig(BaseModel):
    kind: str


class RepeatWrapperConfig(DatasetWrapperConfig):
    repetitions: int = Field(..., ge=2)
    """The number of times to repeat the dataset."""


class ShuffleWrapperConfig(DatasetWrapperConfig):
    seed: int | None = Field(None, ge=0)
    """Random seed for shuffling. If None, a random seed is used."""


class SubsetWrapperConfig(DatasetWrapperConfig):
    indices: Sequence | None = None
    start_index: int | None = None
    end_index: int | None = None
    start_percent: float | None = None
    end_percent: float | None = None


class PipelineConfig(_RegistryBase):
    _registry: ClassVar[dict[str, type]] = {}
    _type_field: ClassVar[str] = "kind"

    kind: str


DatasetWrappers = Union[RepeatWrapperConfig, ShuffleWrapperConfig, SubsetWrapperConfig]

TPipelineConfig = TypeVar("TPipelineConfig", bound=PipelineConfig)


class DatasetBaseConfig[TPipelineConfig: PipelineConfig](_RegistryBase):
    _registry: ClassVar[dict[str, type]] = {}
    _type_field: ClassVar[str] = "kind"

    kind: str | None = None
    """Kind of dataset to use."""
    pipeline: Annotated[TPipelineConfig | None, Discriminated(PipelineConfig)] = Field(None)
    """Config of the pipeline to use for the dataset."""

    dataset_normalizers: (
        dict[
            str,
            list[Annotated[Any, Discriminated(NormalizerConfig)]] | Annotated[Any, Discriminated(NormalizerConfig)],
        ]
        | None
    ) = None

    """List of normalizers to apply to the dataset. The key is the data source name."""
    dataset_wrappers: list[DatasetWrappers] | None = Field(None, validation_alias="wrappers")
    included_properties: set[str] | None = Field(None)
    """Set of properties (i.e., getitem_* methods that are called) of this dataset that will be loaded, if not set all properties are loaded"""
    excluded_properties: set[str] | None = Field(None)
    """Set of properties of this dataset that will NOT be loaded, even if they are present in the included list"""

    model_config = {
        "extra": "forbid",
        "validate_by_name": True,
        "validate_by_alias": True,
    }  # Forbid extra fields in dataset configs


class StandardDatasetConfig(DatasetBaseConfig, ABC):
    """Base config for datasets with fixed splits."""

    root: str
    """Root directory of the dataset."""
    split: Literal["train", "val", "test"]
    """Which split of the dataset to use. Must be one of "train", "val", or "test"."""


class DatasetSplitIDs(BaseModel, ABC):
    """Base class for dataset split ID validation with overlap checking.

    This base class provides:
    1. Automatic validation that train/val/test splits don't have overlapping IDs
    2. Optional size validation for datasets that have expected split sizes

    Subclasses can optionally define class variables for size validation:
    - EXPECTED_TRAIN_SIZE: Expected number of training samples
    - EXPECTED_VAL_SIZE: Expected number of validation samples
    - EXPECTED_TEST_SIZE: Expected number of test samples
    - DATASET_NAME: Name of the dataset for error messages

    If these are not defined, only overlap checking will be performed.
    """

    # Optional - subclasses can define these if they want size validation
    EXPECTED_TRAIN_SIZE: ClassVar[int | None] = None
    EXPECTED_VAL_SIZE: ClassVar[int | None] = None
    EXPECTED_TEST_SIZE: ClassVar[int | None] = None
    EXPECTED_HIDDEN_TEST_SIZE: ClassVar[int | None] = None
    # EXPECTED_EXTRAP_SIZE: ClassVar[int | None] = None
    # EXPECTED_INTERP_SIZE: ClassVar[int | None] = None
    DATASET_NAME: ClassVar[str | None] = None

    train: list[int]
    val: list[int]
    test: list[int]
    extrap: list[int] = []  # Optional OOD extrapolation set
    interp: list[int] = []  # Optional OOD interpolation set
    train_subset: list[int] = []  # Optional subset of training data for logging metrics

    @model_validator(mode="after")
    def validate_splits(self):
        """Validate splits and check for overlaps."""
        # Optional size validation - only if expected sizes are defined
        if self.EXPECTED_TRAIN_SIZE is not None:
            assert len(self.train) == self.EXPECTED_TRAIN_SIZE, (
                f"Train split has length {len(self.train)}. "
                f"Expected {self.EXPECTED_TRAIN_SIZE} for {self.DATASET_NAME}."
            )
        if self.EXPECTED_VAL_SIZE is not None:
            assert len(self.val) == self.EXPECTED_VAL_SIZE, (
                f"Validation split has length {len(self.val)}. "
                f"Expected {self.EXPECTED_VAL_SIZE} for {self.DATASET_NAME}."
            )
        if self.EXPECTED_TEST_SIZE is not None:
            assert len(self.test) == self.EXPECTED_TEST_SIZE, (
                f"Test split has length {len(self.test)}. Expected {self.EXPECTED_TEST_SIZE} for {self.DATASET_NAME}."
            )
        if self.EXPECTED_HIDDEN_TEST_SIZE is not None and hasattr(self, "hidden_test"):
            assert len(self.hidden_test) == self.EXPECTED_HIDDEN_TEST_SIZE, (
                f"Hidden test split has length {len(self.hidden_test)}. "
                f"Expected {self.EXPECTED_HIDDEN_TEST_SIZE} for {self.DATASET_NAME}."
            )

        self._check_no_overlaps()
        return self

    def _check_no_overlaps(self):
        """Check that splits don't have overlapping IDs."""
        # Get all split fields (including any additional ones like hidden_test)
        split_fields = {}
        for field_name in self.__class__.model_fields.keys():
            field_value = getattr(self, field_name)
            if isinstance(field_value, list) and field_value:  # Only check non-empty splits
                split_fields[field_name] = set(field_value)

        # Check all pairs of splits for overlaps. Exclude train_subset from this check.
        field_names = [field_name for field_name in split_fields.keys() if field_name != "train_subset"]
        for i, field1 in enumerate(field_names):
            for field2 in field_names[i + 1 :]:
                overlap = split_fields[field1] & split_fields[field2]
                if overlap:
                    raise ValueError(
                        f"{field1.capitalize()} and {field2} splits have overlapping IDs: {sorted(overlap)}"
                    )
        # Check that train_subset is a subset of training set
        if self.train_subset:
            assert set(self.train_subset).issubset(set(self.train)), "train_subset is not a subset of the training set"


class FieldDimSpec(RootModel[OrderedDict[str, int]]):
    """A specification for a group of named data fields and their dimensions."""

    @property
    def field_slices(self) -> dict[str, slice]:
        """Calculates slice indices for each field in concatenation order."""
        indices = {}
        start = 0
        for field, dim in self.root.items():
            if not isinstance(dim, int) or dim <= 0:
                continue
            indices[field] = slice(start, start + dim)
            start += dim
        return indices

    @property
    def total_dim(self) -> int:
        """Calculates the total dimension of all fields combined."""
        return sum(self.root.values())

    def __getitem__(self, key: str) -> int:
        return self.root[key]

    def __iter__(self):
        return iter(self.root.items())

    def __getattr__(self, name: str) -> int:
        """Enables attribute-style access (e.g., `spec.geometry`)."""
        try:
            return self.root[name]
        except KeyError as err:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from err

    def __dir__(self) -> list[str]:
        """Improves autocompletion for dynamic attributes."""
        return sorted(set(super().__dir__()) | set(self.root.keys()))

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()

    def items(self):
        return self.root.items()


class DomainDataSpec(BaseModel):
    """Data specification for a single domain (e.g., surface, volume, wake)."""

    output_dims: FieldDimSpec
    """Output fields and their dimensions for this domain, e.g. {"pressure": 1, "velocity": 3}."""
    feature_dim: FieldDimSpec | None = None
    """Input feature fields and their dimensions for this domain."""


class ModelDataSpecs(BaseModel):
    """Base data specification for models that operate on arbitrary named domains.

    This is the minimal interface that model configs need from data specifications:
    position dimensions, available conditioning, and per-domain data descriptions.
    """

    position_dim: int = Field(..., ge=1)
    """Dimension of the input position vectors."""
    conditioning_dims: FieldDimSpec | None = None
    """Available conditioning features and their dimensions."""
    domains: dict[str, DomainDataSpec] = Field(default_factory=dict)
    """Per-domain data specifications keyed by domain name."""
    use_physics_features: bool = False
    """Whether physics features are used as input."""

    @property
    def total_output_dim(self) -> int:
        """Calculates the total output dimension across all domains."""
        return sum(spec.output_dims.total_dim for spec in self.domains.values())

    @property
    def all_targets(self) -> set[str]:
        """Returns all target field names across all domains, prefixed by domain name."""
        targets: set[str] = set()
        for name, spec in self.domains.items():
            targets |= {f"{name}_{key}" for key in spec.output_dims.keys()}
        return targets

    @property
    def all_features(self) -> set[str]:
        """Returns all feature field names across all domains."""
        features: set[str] = set()
        for spec in self.domains.values():
            if spec.feature_dim:
                features |= set(spec.feature_dim.keys())
        return features

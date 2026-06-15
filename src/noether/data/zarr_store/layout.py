#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Mapping between Noether ``FileMap`` fields and the per-field Zarr array layout.

Every CFD field is assigned to a domain (``surface`` / ``volume``), a canonical
name (``f"{domain}_{name}"`` — matching the dataset's ``getitem_*`` / normalizer
keys), a channel width, and a *kind*:

* ``coord`` — point positions, stored float32;
* ``value`` — physical quantities, stored float16.

Each field becomes its own Zarr array (``<domain>/<name>``), so fields are read
independently; all arrays of a domain share the shuffle permutation and chunk grid.
"""

from __future__ import annotations

from dataclasses import dataclass

from noether.data.schemas import FileMap
from noether.data.zarr_store.manifest import ArrayLayout, DomainLayout


@dataclass(frozen=True)
class FieldSpec:
    """Static description of one CFD field."""

    filemap_attr: str
    """Attribute name on :class:`~noether.data.schemas.FileMap`."""
    domain: str
    name: str
    """Short name within the domain; combined as ``f"{domain}_{name}"``."""
    dim: int
    """Channel width (1 for scalars, 3 for vectors)."""
    kind: str
    """``"coord"`` or ``"value"``."""

    @property
    def canonical(self) -> str:
        """Canonical field key, e.g. ``"volume_velocity"``."""
        return f"{self.domain}_{self.name}"


# Per-sample metadata that is not a point cloud and therefore not stored as point arrays.
EXCLUDED_FILEMAP_FIELDS: frozenset[str] = frozenset({"design_parameters"})

# FileMap attributes whose (domain, name) cannot be derived by splitting on the first
# underscore: the STL point clouds have their own point counts, so they get their own
# domains (a domain = one aligned point set with one chunk grid).
_DOMAIN_OVERRIDES: dict[str, tuple[str, str]] = {
    "surface_position_stl": ("surface_stl", "position"),
    "surface_position_stl_resampled": ("surface_stl_resampled", "position"),
}

# Short-name overrides so canonical keys match the dataset getitem_* / normalizer keys.
_NAME_OVERRIDES: dict[str, str] = {"distance_to_surface": "sdf"}

# Channel width per semantic field name (scalars 1, vectors 3).
_FIELD_DIMS: dict[str, int] = {
    "position": 3,
    "pressure": 1,
    "friction": 3,
    "normals": 3,
    "area": 1,
    "velocity": 3,
    "vorticity": 3,
    "sdf": 1,
}


def _build_field_specs() -> list[FieldSpec]:
    """Derive one :class:`FieldSpec` per point-cloud field of the ``FileMap`` schema.

    Generic over all CFD datasets: any ``FileMap`` attribute (current or future) must be
    either excluded here or resolvable to a (domain, name, dim) — unknown names raise at
    import time so new schema fields cannot be silently dropped from conversion.
    """
    specs: list[FieldSpec] = []
    for attr in FileMap.model_fields:
        if attr in EXCLUDED_FILEMAP_FIELDS:
            continue
        domain, _, name = attr.partition("_")
        domain, name = _DOMAIN_OVERRIDES.get(attr, (domain, name))
        name = _NAME_OVERRIDES.get(name, name)
        if name not in _FIELD_DIMS:
            raise ValueError(
                f"FileMap field '{attr}' has no dim/kind mapping in noether.data.zarr_store.layout; "
                "add it to _FIELD_DIMS (or EXCLUDED_FILEMAP_FIELDS if it is not a point cloud)."
            )
        kind = "coord" if name == "position" else "value"
        specs.append(FieldSpec(attr, domain, name, _FIELD_DIMS[name], kind))
    return specs


# Order follows the FileMap schema; it defines the array order inside each domain layout.
# Positions (kind="coord") are stored float32; everything else float16 by default.
FIELD_SPECS: list[FieldSpec] = _build_field_specs()

CANONICAL_TO_SPEC: dict[str, FieldSpec] = {spec.canonical: spec for spec in FIELD_SPECS}


def present_specs(filemap: FileMap) -> list[FieldSpec]:
    """Return the :data:`FIELD_SPECS` whose ``filemap_attr`` is set on *filemap*."""
    return [spec for spec in FIELD_SPECS if getattr(filemap, spec.filemap_attr, None) is not None]


def build_domain_layouts(
    filemap: FileMap,
    coords_dtype: str = "float32",
    values_dtype: str = "float16",
    field_dtypes: dict[str, str] | None = None,
) -> dict[str, DomainLayout]:
    """Build the per-domain :class:`DomainLayout` (one array per field) present in *filemap*.

    Args:
        filemap: Field-to-filename mapping for the dataset being converted.
        coords_dtype: Dtype string for position arrays.
        values_dtype: Dtype string for physical field arrays.
        field_dtypes: Per-field dtype overrides keyed by canonical name, e.g.
            ``{"volume_vorticity": "float32"}`` for fields whose dynamic range exceeds
            ``values_dtype`` (float16 caps at ~6.6e4).

    Returns:
        Mapping ``domain -> DomainLayout``.

    Raises:
        ValueError: If a domain has no (or more than one) coordinate field.
    """
    specs = present_specs(filemap)
    domains = sorted({spec.domain for spec in specs})
    overrides = field_dtypes or {}

    layouts: dict[str, DomainLayout] = {}
    for domain in domains:
        domain_specs = [s for s in specs if s.domain == domain]
        coord_specs = [s for s in domain_specs if s.kind == "coord"]
        if len(coord_specs) != 1:
            raise ValueError(f"Domain '{domain}' must have exactly one coordinate field, found {len(coord_specs)}.")
        arrays = {
            spec.canonical: ArrayLayout(
                array_name=f"{domain}/{spec.name}",
                field=spec.canonical,
                dtype=overrides.get(spec.canonical, coords_dtype if spec.kind == "coord" else values_dtype),
                dim=spec.dim,
            )
            for spec in domain_specs
        }
        layouts[domain] = DomainLayout(position=coord_specs[0].canonical, arrays=arrays)
    return layouts


def filename_to_canonical(filemap: FileMap) -> dict[str, str]:
    """Reverse map ``stored_filename -> canonical_field`` for the present fields.

    Lets a Zarr-backed dataset translate the ``.pt`` filename referenced by an
    ``AeroDataset.getitem_*`` method back to the canonical field key used in the
    Zarr store.
    """
    mapping: dict[str, str] = {}
    for spec in present_specs(filemap):
        filename = getattr(filemap, spec.filemap_attr)
        mapping[filename] = spec.canonical
    return mapping

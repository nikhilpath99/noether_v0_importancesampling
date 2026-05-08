#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from noether.core.schemas.dataset import DomainDataSpec, ModelDataSpecs
from noether.core.schemas.normalizers import FieldNormalizerConfig

from .base import AeroCFDPreset, AeroPipelineParams


class AirFRANSPreset(AeroCFDPreset):
    """Preset for the AirFRANS 2D RANS airfoil dataset.

    Supports both per-simulation ``.pt`` files and the mmap cache produced by
    ``build_cache.py``.  Pass ``cache_dir`` at construction time to enable the
    cache backend — no VTK parsing or ``airfrans`` library required at training
    time.

    Dataset splits:
        - ``reynolds_train`` / ``reynolds_test`` — Reynolds-number extrapolation
        - ``full_train`` / ``full_test`` — standard random split
        - ``scarce_train`` — low-data regime (200 samples)
        - ``aoa_train`` / ``aoa_test`` — angle-of-attack extrapolation

    Example::

        preset = AirFRANSPreset(cache_dir="/data/airfrans/cache")
        config = preset.build_config(
            model_kind="noether.modeling.models.aerodynamics.AeroABUPT",
            model_params=dict(hidden_dim=128, geometry_depth=4,
                              physics_blocks=["perceiver"] + ["shared", "cross"] * 3),
            trainer_kind="noether.training.trainers.WeightedLossTrainer",
            trainer_params=dict(field_weights=FIELD_WEIGHTS),
            dataset_root="/data/airfrans",
            datasets={"train": "reynolds_train", "test": "reynolds_test"},
            max_epochs=200,
            accelerator="gpu",
        )
    """

    dataset_kind = "noether.data.datasets.cfd.AirFRANSDataset"

    # Statistics computed from the full_train split (800 simulations).
    # Position units: metres.  Pressure: Pascals.  Velocity: m/s.
    stats: dict = {
        # Positions
        "surface_pos_mean":             [0.3831, 0.0079],
        "surface_pos_std":              [0.3958, 0.0334],
        "volume_pos_mean":              [0.2844, 0.0112],
        "volume_pos_std":               [0.5962, 0.3184],
        "raw_pos_min":                  [-2.1648, -1.6456],
        "raw_pos_max":                  [4.2321,   1.6213],
        # Pressure
        "surface_pressure_mean":        [-1154.78],
        "surface_pressure_std":         [4892.50],
        "volume_pressure_mean":         [-448.14],
        "volume_pressure_std":          [2931.82],
        # Velocity (volume only)
        "volume_velocity_mean":         [15.6313,  8.6995],
        "volume_velocity_std":          [31.6532,  28.531],
        # Design parameters: [velocity, aoa_deg, naca_p1, p2, p3, p4]
        "inflow_design_parameters_mean": [62.116, 4.5022, 2.6836, 4.3984, 6.0924, 6.5902],
        "inflow_design_parameters_std":  [17.914, 5.4403, 1.7915, 2.2238, 6.5746, 6.9615],
    }

    # AirFRANS has ~1 011 surface points and ~179 K volume points per sim.
    # Point counts are scaled down accordingly relative to 3-D datasets.
    pipeline_defaults: AeroPipelineParams = {
        "num_surface_points": 1024,
        "num_volume_points":  4096,
        "num_surface_queries": 0,
        "num_volume_queries":  0,
        "use_physics_features": False,
    }

    pipeline_model_overrides: dict[str, AeroPipelineParams] = {
        "noether.modeling.models.aerodynamics.AeroABUPT": {
            "num_geometry_supernodes":    256,
            "num_geometry_points":       4096,
            "num_surface_anchor_points":  128,
            "num_volume_anchor_points":   256,
            "num_surface_queries": 0,
            "num_volume_queries":  0,
        },
    }

    # AB-UPT forward pass needs design_parameters alongside geometry inputs.
    forward_properties_map: dict[str, list[str]] = {
        "noether.modeling.models.aerodynamics.AeroABUPT": [
            "geometry_position",
            "geometry_supernode_idx",
            "geometry_batch_idx",
            "surface_anchor_position",
            "volume_anchor_position",
            "design_parameters",
        ],
        "_default": [
            "surface_position",
            "volume_position",
            "surface_features",
            "volume_features",
        ],
    }

    def __init__(self, cache_dir: str | None = None) -> None:
        """
        Args:
            cache_dir: Optional path to the directory holding ``full_train.cache/``
                and ``full_test.cache/`` subdirectories.  When set the dataset reads
                from memory-mapped arrays instead of per-simulation ``.pt`` files.
        """
        self.cache_dir = cache_dir

    @property
    def data_specs(self) -> ModelDataSpecs:
        return ModelDataSpecs(
            position_dim=2,
            domains={
                "surface": DomainDataSpec(output_dims={"pressure": 1}),
                "volume":  DomainDataSpec(output_dims={"pressure": 1, "velocity": 2}),
            },
            conditioning_dims={"design_parameters": 6},
        )

    @property
    def normalizer_spec(self) -> dict[str, FieldNormalizerConfig]:
        return {
            "surface_pressure": FieldNormalizerConfig(strategy="mean_std"),
            "volume_pressure":  FieldNormalizerConfig(strategy="mean_std"),
            "volume_velocity":  FieldNormalizerConfig(strategy="mean_std"),
            "design_parameters": FieldNormalizerConfig(
                strategy="mean_std",
                stat_keys={
                    "mean": "inflow_design_parameters_mean",
                    "std":  "inflow_design_parameters_std",
                },
            ),
            "surface_position": FieldNormalizerConfig(
                strategy="position",
                scale=1,
                stat_keys={"min": "raw_pos_min", "max": "raw_pos_max"},
            ),
            "volume_position": FieldNormalizerConfig(
                strategy="position",
                scale=1,
                stat_keys={"min": "raw_pos_min", "max": "raw_pos_max"},
            ),
        }

    @property
    def excluded_properties(self) -> set[str]:
        return {
            "surface_friction",
            "surface_normals",
            "volume_normals",
            "volume_vorticity",
            "volume_sdf",
            "surface_sdf",
            "volume_importance_weights",
            "surface_importance_weights",
        }

    def target_properties(self) -> list[str]:
        return [
            "surface_pressure_target",
            "volume_pressure_target",
            "volume_velocity_target",
        ]

    def build_dataset(
        self,
        *,
        split: str,
        root: str,
        model_kind: str,
        wrappers=None,
        **overrides,
    ):
        """Build an ``AirFRANSDatasetConfig``, wiring in the cache if available.

        Args:
            split: AirFRANS split name (e.g. ``"reynolds_train"``).
            root: Directory containing ``manifest.json``.
            model_kind: Fully-qualified model class path (determines pipeline overrides).
            wrappers: Optional dataset wrapper list.
            **overrides: Pipeline parameter overrides forwarded to ``build_pipeline``.
        """
        from noether.data.datasets.cfd.airfrans.dataset import AirFRANSDatasetConfig

        return AirFRANSDatasetConfig(
            root=root,
            split=split,
            cache_dir=self.cache_dir,
            pipeline=self.build_pipeline(model_kind, **overrides),
            dataset_normalizers=self.build_normalizers(),
            dataset_wrappers=wrappers,
            excluded_properties=self.excluded_properties,
        )

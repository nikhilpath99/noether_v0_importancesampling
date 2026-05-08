#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Train aerodynamic models on the AirFRANS 2D RANS airfoil dataset.

Usage (AB-UPT, Reynolds extrapolation split)::

    python -m aero_cfd.scripts.train_airfrans \\
        --dataset-root /data/airfrans \\
        --cache-dir    /data/airfrans/cache \\
        --output-path  /data/airfrans/outputs \\
        --split        reynolds \\
        --model        abupt

The ``--cache-dir`` flag enables memory-mapped loading from the
``full_train.cache/`` / ``full_test.cache/`` directories built by
``build_cache.py`` — no VTK parsing or ``airfrans`` library required.
"""

from aero_cfd.presets import AirFRANSPreset
from noether.core.distributed.utils import accelerator_to_device
from noether.training.runners import HydraRunner

TRAINER_KIND = "noether.training.trainers.WeightedLossTrainer"

FIELD_WEIGHTS = {
    "surface_pressure": 1.0,
    "volume_pressure":  1.0,
    "volume_velocity":  1.0,
}

# Map CLI split alias → (train_split, test_split)
SPLIT_MAP = {
    "reynolds": ("reynolds_train", "reynolds_test"),
    "full":     ("full_train",     "full_test"),
    "aoa":      ("aoa_train",      "aoa_test"),
    "scarce":   ("scarce_train",   "full_test"),
}


def train_abupt(
    *,
    dataset_root: str,
    output_path: str,
    cache_dir: str | None = None,
    split: str = "reynolds",
    accelerator: str = "gpu",
    num_workers: int = 0,
    max_epochs: int = 200,
    hidden_dim: int = 128,
) -> None:
    """Train AB-UPT on AirFRANS.

    Args:
        dataset_root: Directory containing ``manifest.json``.
        output_path: Training output directory (checkpoints, logs).
        cache_dir: Optional path to the mmap cache directory.
        split: Split alias — one of ``reynolds``, ``full``, ``aoa``, ``scarce``.
        accelerator: ``"gpu"``, ``"cpu"``, or ``"mps"``.
        num_workers: DataLoader worker processes (use 0 when CPU is constrained).
        max_epochs: Training epochs.
        hidden_dim: AB-UPT hidden dimension.
    """
    train_split, test_split = SPLIT_MAP[split]
    preset = AirFRANSPreset(cache_dir=cache_dir)

    config = preset.build_config(
        model_kind="noether.modeling.models.aerodynamics.AeroABUPT",
        model_params=dict(
            hidden_dim=hidden_dim,
            geometry_depth=4,
            physics_blocks=["perceiver"] + ["shared", "cross"] * 3,
        ),
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS),
        num_workers=num_workers,
        dataset_root=dataset_root,
        output_path=output_path,
        datasets={"train": train_split, "test": test_split},
        max_epochs=max_epochs,
        batch_size=4,
        accelerator=accelerator,
    )
    HydraRunner().main(device=accelerator_to_device(accelerator), config=config)


MODELS = {
    "abupt": train_abupt,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train aerodynamic models on AirFRANS.")
    parser.add_argument("--dataset-root", required=True,
                        help="Directory containing manifest.json.")
    parser.add_argument("--output-path", required=True,
                        help="Training output directory.")
    parser.add_argument("--cache-dir", default=None,
                        help="Path to mmap cache directory (optional).")
    parser.add_argument("--split", default="reynolds",
                        choices=list(SPLIT_MAP),
                        help="Train/test split alias.")
    parser.add_argument("--model", default="abupt",
                        choices=list(MODELS),
                        help="Model architecture.")
    parser.add_argument("--accelerator", default="gpu",
                        choices=["cpu", "gpu", "mps"])
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 = main process only).")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    MODELS[args.model](
        dataset_root=args.dataset_root,
        output_path=args.output_path,
        cache_dir=args.cache_dir,
        split=args.split,
        accelerator=args.accelerator,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        hidden_dim=args.hidden_dim,
    )

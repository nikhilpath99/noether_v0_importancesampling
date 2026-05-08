#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""AirFRANS dataset — 2D RANS airfoil simulations."""

import json
import logging
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import torch

from noether.core.schemas.dataset import DatasetBaseConfig
from noether.core.utils.common import validate_path
from noether.data.datasets.cfd.airfrans.filemap import AIRFRANS_FILEMAP
from noether.data.datasets.cfd.dataset import AeroDataset

logger = logging.getLogger(__name__)

AirFRANSSplit = Literal[
    "full_train",
    "scarce_train",
    "reynolds_train",
    "aoa_train",
    "full_test",
    "reynolds_test",
    "aoa_test",
]

# Cache subdirectory names produced by Eigen_Decomposition build_cache.py
_CACHE_SUBDIRS = [
    ("train", "full_train.cache"),
    ("test", "full_test.cache"),
]


class AirFRANSDatasetConfig(DatasetBaseConfig):
    """Configuration for the AirFRANS dataset.

    Args:
        root: Root directory containing the manifest.json file and, when not
            using a cache, the converted per-simulation ``.pt`` folders.
        split: Which manifest split to load. Use ``reynolds_train`` /
            ``reynolds_test`` for the Reynolds-number extrapolation benchmark.
        manifest: Filename of the manifest inside *root* (default:
            ``manifest.json``).
        cache_dir: Optional path to a directory that holds the mmap cache
            subdirectories ``full_train.cache/`` and ``full_test.cache/``
            produced by ``build_cache.py``.  When set the dataset reads
            directly from those memory-mapped arrays instead of per-simulation
            ``.pt`` files — no VTK parsing or ``airfrans`` library required.

    Example:

        .. testcode::

            cfg = AirFRANSDatasetConfig(
                root="/data/airfrans",
                split="reynolds_train",
                cache_dir="/data/airfrans/cache",
            )
    """

    kind: str | None = "noether.data.datasets.cfd.AirFRANSDataset"
    root: str
    split: AirFRANSSplit
    manifest: str = "manifest.json"
    cache_dir: str | None = None


class AirFRANSDataset(AeroDataset):
    """Dataset for AirFRANS 2D RANS airfoil simulations.

    Supports two loading modes:

    **Per-simulation .pt files** (default)
        Each sample directory contains per-field tensors produced by the
        ``convert_airfrans`` preprocessing tool.

    **mmap cache** (set ``cache_dir`` in config)
        Reads from the concatenated ``.npy`` cache built by
        ``build_cache.py`` in the Eigen_Decomposition project.  All arrays
        are memory-mapped so only the requested slices are paged in.

    Positions are 2-D (x, y); the z coordinate is discarded during
    preprocessing.

    Fields per sample:

    - ``volume_position``   (N, 2)
    - ``volume_velocity``   (N, 2)  — (U_x, U_y)
    - ``volume_pressure``   (N, 1)
    - ``surface_position``  (M, 2)
    - ``surface_pressure``  (M, 1)
    - ``surface_normals``   (M, 2)
    - ``design_parameters`` (6,)  — [velocity, aoa_deg, p1, p2, p3, p4]
      where p4 = 0 for NACA 4-digit airfoils

    Args:
        dataset_config: Dataset configuration including root path and split.
        filemap: Optional custom filemap (defaults to
            :data:`AIRFRANS_FILEMAP`).
    """

    STATS_FILE: ClassVar[str] = str(Path(__file__).parent / "stats.yaml")

    def __init__(
        self,
        dataset_config: AirFRANSDatasetConfig,
        filemap=AIRFRANS_FILEMAP,
    ):
        """
        Args:
            dataset_config: Configuration for the dataset.
            filemap: FileMap defining field-to-filename mapping.

        Raises:
            ValueError: If the requested split is not present in the manifest.
            FileNotFoundError: If the root directory or manifest does not exist.
        """
        super().__init__(dataset_config=dataset_config, filemap=filemap)

        self.split = dataset_config.split
        self.source_root = validate_path(dataset_config.root)

        manifest_path = self.source_root / dataset_config.manifest
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        if self.split not in manifest:
            raise ValueError(
                f"Split '{self.split}' not found in manifest. "
                f"Available splits: {list(manifest.keys())}"
            )

        self.design_ids: list[str] = manifest[self.split]

        # Cache mode — mmap the concatenated .npy arrays
        self._cache: dict[str, dict] | None = None
        self._name_to_cache: dict[str, tuple[str, int]] | None = None
        if dataset_config.cache_dir is not None:
            self._init_cache(Path(dataset_config.cache_dir))

        logger.info(
            "Initialized AirFRANSDataset split='%s' with %d samples from %s%s",
            self.split,
            len(self.design_ids),
            self.source_root,
            " (cache mode)" if self._cache is not None else "",
        )

    def _init_cache(self, cache_dir: Path) -> None:
        """Open mmap handles for all available cache subdirectories.

        Args:
            cache_dir: Directory containing ``full_train.cache/`` and/or
                ``full_test.cache/`` subdirectories.

        Raises:
            FileNotFoundError: If no cache subdirectories are found.
        """
        self._cache = {}
        self._name_to_cache = {}

        for key, subdir in _CACHE_SUBDIRS:
            cache_path = cache_dir / subdir
            if not cache_path.is_dir():
                continue
            meta = np.load(cache_path / "meta.npz", allow_pickle=False)
            names = [str(n) for n in meta["names"]]
            self._cache[key] = {
                "points": np.load(cache_path / "points.npy", mmap_mode="r"),
                "inputs": np.load(cache_path / "inputs.npy", mmap_mode="r"),
                "fields": np.load(cache_path / "fields.npy", mmap_mode="r"),
                "surface": np.load(cache_path / "surface.npy", mmap_mode="r"),
                "offsets": meta["offsets"],
                "inlet_vel": meta["inlet_vel"],
                "aoa": meta["aoa"],
                "naca": meta["naca"],
            }
            for local_idx, name in enumerate(names):
                self._name_to_cache[name] = (key, local_idx)

        if not self._cache:
            raise FileNotFoundError(
                f"No cache subdirectories found in {cache_dir}. "
                f"Expected: {[s for _, s in _CACHE_SUBDIRS]}"
            )

    def __len__(self) -> int:
        return len(self.design_ids)

    def _load_from_disk(self, idx: int, filename: str) -> torch.Tensor:
        """Load a tensor from the per-simulation .pt directory.

        Args:
            idx: Sample index.
            filename: Tensor filename (e.g. ``volume_position.pt``).

        Returns:
            Loaded tensor.

        Raises:
            RuntimeError: If the file cannot be loaded.
        """
        path = self.source_root / self.design_ids[idx] / filename
        try:
            return torch.load(path, weights_only=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}") from e

    def _load_from_cache(self, idx: int, filename: str) -> torch.Tensor:
        """Load a field by slicing the mmap'd cache arrays.

        The cache stores all simulations concatenated in ``points``,
        ``inputs``, ``fields``, and ``surface`` arrays.  Per-simulation
        slices are recovered via the ``offsets`` vector in ``meta.npz``.

        Cache layout (columns):
            - ``inputs``  [:, 0:2] — inlet velocity components (U_x, U_y)
            - ``inputs``  [:, 2]   — signed distance function
            - ``inputs``  [:, 3:5] — surface normals (n_x, n_y)
            - ``fields``  [:, 0:2] — velocity (U_x, U_y)
            - ``fields``  [:, 2]   — pressure
            - ``fields``  [:, 3]   — turbulent viscosity ν_t
            - ``surface`` [:, 0]   — bool mask (True = surface point)

        Args:
            idx: Sample index.
            filename: Filemap filename string identifying the requested field.

        Returns:
            Tensor with the same shape as the corresponding ``.pt`` file.

        Raises:
            KeyError: If *filename* is not a known cache field.
        """
        name = self.design_ids[idx]
        cache_key, local_idx = self._name_to_cache[name]
        c = self._cache[cache_key]

        s = int(c["offsets"][local_idx])
        e = int(c["offsets"][local_idx + 1])

        srf = c["surface"][s:e, 0]          # (P,) bool
        pts = np.array(c["points"][s:e])     # copy to avoid mmap lifetime issues
        inp = np.array(c["inputs"][s:e])
        fld = np.array(c["fields"][s:e])

        fm = self.filemap
        if filename == fm.volume_position:
            return torch.from_numpy(pts[~srf])
        elif filename == fm.volume_velocity:
            return torch.from_numpy(fld[~srf, :2])
        elif filename == fm.volume_pressure:
            return torch.from_numpy(fld[~srf, 2])   # (N,) — getitem unsqueezes to (N,1)
        elif filename == fm.surface_position:
            return torch.from_numpy(pts[srf])
        elif filename == fm.surface_pressure:
            return torch.from_numpy(fld[srf, 2])    # (M,) — getitem unsqueezes to (M,1)
        elif filename == fm.surface_normals:
            return torch.from_numpy(inp[srf, 3:5])
        elif filename == fm.design_parameters:
            naca = c["naca"][local_idx]      # (4,) float32
            params = np.array(
                [c["inlet_vel"][local_idx], c["aoa"][local_idx],
                 naca[0], naca[1], naca[2], naca[3]],
                dtype=np.float32,
            )
            return torch.from_numpy(params)
        else:
            raise KeyError(f"Unknown field '{filename}' for cache-backed AirFRANSDataset")

    def _load(self, idx: int, filename: str) -> torch.Tensor:
        if self._cache is not None:
            return self._load_from_cache(idx, filename)
        return self._load_from_disk(idx, filename)

    def sample_info(self, idx: int) -> dict:
        """Return metadata about sample *idx*."""
        name = self.design_ids[idx]
        return {
            "sample_uri": self.source_root / name,
            "sim_name": name,
            "split": self.split,
        }

"""Filesystem helpers shared across dataset modules."""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


def resolve_cache_dir(cache_dir: str | Path | None, *, default_name: str) -> Path:
    """Return a writable cache directory, creating it if needed."""
    base = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / default_name
    base.mkdir(parents=True, exist_ok=True)
    return base


def ensure_file(path: Path, url: str) -> Path:
    """Download ``url`` into ``path`` if it doesn't already exist."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def ensure_csv(
    cache_dir: Path,
    filename: str,
    url: str,
    data_path: Optional[str | Path] = None,
) -> Path:
    """Resolve a CSV file via cache download or user-provided path."""
    if data_path is not None:
        return Path(data_path)
    target = cache_dir / filename
    if not target.exists():
        urllib.request.urlretrieve(url, target)
    return target


def download_archive(url: str, path: Path) -> Path:
    """Download an archive to ``path`` if needed."""
    return ensure_file(path, url)


def ensure_tar_extracted(archive: Path, extract_root: Path, target_dir: str) -> Path:
    """Extract a tar.gz archive into ``extract_root`` and return the target dir."""
    target_path = extract_root / target_dir
    if target_path.exists():
        return target_path
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=extract_root)
    return target_path


def to_host_jax_array(array: np.ndarray) -> jax.Array:
    """Copy a NumPy array onto the default CPU device for JAX consumption."""
    cpu_devices = jax.devices("cpu")
    if cpu_devices:
        with jax.default_device(cpu_devices[0]):
            return jnp.asarray(array)
    return jnp.asarray(array)

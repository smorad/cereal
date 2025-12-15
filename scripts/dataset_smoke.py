#!/usr/bin/env python3
"""Ad-hoc script for manually exercising dataset download + disk-source flows."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Type

from cyreal.datasets import (
    CIFAR10Dataset,
    CIFAR100Dataset,
    DailyMinTemperaturesDataset,
    EMNISTDataset,
    FashionMNISTDataset,
    KMNISTDataset,
    MNISTDataset,
    SunspotsDataset,
)
from cyreal.datasets.utils import resolve_cache_dir

DatasetCls = Type[MNISTDataset]
DatasetInfo = Tuple[DatasetCls, dict, dict]

DATASETS: Dict[str, DatasetInfo] = {
    "mnist": (MNISTDataset, {}, {}),
    "fashion-mnist": (FashionMNISTDataset, {}, {}),
    "kmnist": (KMNISTDataset, {}, {}),
    "emnist-letters": (EMNISTDataset, {"subset": "letters"}, {"subset": "letters"}),
    "cifar10": (CIFAR10Dataset, {}, {}),
    "cifar100": (CIFAR100Dataset, {}, {}),
    "daily-min": (DailyMinTemperaturesDataset, {}, {}),
    "sunspots": (SunspotsDataset, {}, {}),
}


def _format_shapes(batch) -> dict[str, tuple[int, ...]]:
    return {key: tuple(int(dim) for dim in value.shape) for key, value in batch.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS.keys()),
        default=sorted(DATASETS.keys()),
        help="Datasets to run smoke checks for.",
    )
    parser.add_argument("--split", choices=["train", "test"], default="train", help="Dataset split to inspect.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Base directory for caching dataset artifacts (defaults to dataset defaults).",
    )
    parser.add_argument(
        "--prefetch-size",
        type=int,
        default=64,
        help="Prefetch size to use when constructing disk sources.",
    )
    parser.add_argument(
        "--ordering",
        choices=["sequential", "shuffle"],
        default="sequential",
        help="Sample ordering for disk sources.",
    )
    return parser.parse_args()


def smoke_dataset(dataset_name: str, info: DatasetInfo, args: argparse.Namespace) -> None:
    dataset_cls, init_kwargs, disk_kwargs = info
    cache_dir = None
    if args.cache_dir is not None:
        target = Path(args.cache_dir).expanduser().resolve() / dataset_name
        cache_dir = resolve_cache_dir(target, default_name=dataset_name)

    init_params = dict(init_kwargs)
    init_params.setdefault("split", args.split)
    if cache_dir is not None:
        init_params["cache_dir"] = cache_dir

    print(f"[{dataset_name}] Loading in-memory dataset (split={init_params['split']}).")
    dataset = dataset_cls(**init_params)
    first_sample = dataset[0]
    print(f"  len={len(dataset)} sample_shapes={_format_shapes(first_sample)}")

    disk_params = dict(disk_kwargs)
    disk_params.setdefault("split", args.split)
    disk_params.setdefault("ordering", args.ordering)
    disk_params.setdefault("prefetch_size", args.prefetch_size)
    if cache_dir is not None:
        disk_params["cache_dir"] = cache_dir

    print(
        f"[{dataset_name}] Building disk source (split={disk_params['split']}, ordering={disk_params['ordering']}, "
        f"prefetch={disk_params['prefetch_size']})."
    )
    source = dataset_cls.make_disk_source(**disk_params)
    spec = source.element_spec()
    spec_shapes = {key: tuple(int(dim) for dim in value.shape) for key, value in spec.items()}
    print(f"  sample_spec={spec_shapes}")


def main() -> None:
    args = parse_args()
    for name in args.datasets:
        try:
            smoke_dataset(name, DATASETS[name], args)
        except Exception as exc:  # noqa: BLE001 - we want a simple CLI
            print(f"[{name}] FAILED: {exc}")
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

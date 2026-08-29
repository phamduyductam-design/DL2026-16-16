#!/usr/bin/env python3
"""Validate MVTec AD and create the frozen normal-validation manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_protocol import (  # noqa: E402
    MVTECADCatalog,
    assert_protocol_integrity,
    build_evaluation_manifest,
    build_manifest,
    deterministic_normal_split,
    file_checksum_inventory,
    load_config,
    resolve_data_root,
    verify_image_files,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/data_protocol_week1.yaml",
        help="JSON-compatible YAML protocol config",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="MVTec AD root; otherwise read from the configured environment variable",
    )
    parser.add_argument(
        "--output",
        default="outputs/data_protocol/split_seed42.json",
        help="Destination split manifest",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Decode and verify every image and ground-truth mask",
    )
    parser.add_argument(
        "--hash-audit",
        action="store_true",
        help="Audit duplicate images and write a SHA-256 sidecar for all selected files",
    )
    parser.add_argument(
        "--checksum-output",
        default=None,
        help="SHA-256 inventory path; defaults next to --output when --hash-audit is set",
    )
    parser.add_argument(
        "--evaluation-output",
        default=None,
        help="Evaluation-only test annotation manifest; defaults next to --output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    data_root = resolve_data_root(args.data_root, config.data_root_env)

    official_train, official_test = MVTECADCatalog(data_root, config.categories).scan()
    train, validation = deterministic_normal_split(
        official_train,
        config.validation_fraction,
        config.seed,
    )
    assert_protocol_integrity(
        train,
        validation,
        official_test,
        hash_audit=args.hash_audit,
    )
    if args.verify_images:
        verify_image_files([*train, *validation, *official_test])

    transform = {
        "resolution": [config.transform.height, config.transform.width],
        "mean": list(config.transform.mean),
        "std": list(config.transform.std),
        "horizontal_flip_probability": config.transform.horizontal_flip_probability,
        "seed": config.transform.seed,
    }
    manifest = build_manifest(
        data_root,
        train,
        validation,
        official_test,
        seed=config.seed,
        validation_fraction=config.validation_fraction,
        source=config.source,
        transform=transform,
    )
    destination = write_manifest(manifest, args.output)
    manifest_path = Path(args.output)
    evaluation_output = args.evaluation_output or str(
        manifest_path.with_name(f"{manifest_path.stem}_evaluation_only.json")
    )
    evaluation_destination = write_manifest(
        build_evaluation_manifest(data_root, official_test, source=config.source),
        evaluation_output,
    )
    checksum_destination: Path | None = None
    if args.hash_audit:
        checksum_output = args.checksum_output
        if checksum_output is None:
            checksum_output = str(
                manifest_path.with_name(f"{manifest_path.stem}_checksums_sha256.json")
            )
        checksum_destination = write_manifest(
            file_checksum_inventory(data_root, config.categories),
            checksum_output,
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "data_root": str(data_root),
                "manifest": str(destination),
                "evaluation_manifest": str(evaluation_destination),
                "checksum_manifest": (
                    str(checksum_destination) if checksum_destination else None
                ),
                **manifest["counts"],
                "test_anomaly": sum(record.is_anomaly for record in official_test),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

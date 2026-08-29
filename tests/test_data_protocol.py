from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from data_protocol import (
    MVTECADCatalog,
    MVTECDataset,
    MVTecTransform,
    ProtocolError,
    TransformConfig,
    assert_protocol_integrity,
    build_evaluation_manifest,
    build_manifest,
    deterministic_normal_split,
    file_checksum_inventory,
    verify_image_files,
)


class DataProtocolTests(unittest.TestCase):
    categories = ("wood", "metal_nut", "capsule")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self._pixel_counter = 1
        for category in self.categories:
            for index in range(10):
                self._write_image(self.root / category / "train" / "good" / f"{index:03}.png")
            for index in range(2):
                self._write_image(self.root / category / "test" / "good" / f"{index:03}.png")
            for index in range(3):
                self._write_image(self.root / category / "test" / "scratch" / f"{index:03}.png")
                self._write_mask(
                    self.root
                    / category
                    / "ground_truth"
                    / "scratch"
                    / f"{index:03}_mask.png"
                )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        value = self._pixel_counter % 255
        self._pixel_counter += 1
        Image.new("RGB", (12, 8), color=(value, value // 2, 255 - value)).save(path)

    @staticmethod
    def _write_mask(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("L", (12, 8), color=0)
        for x in range(3, 8):
            for y in range(2, 6):
                image.putpixel((x, y), 255)
        image.save(path)

    def _scan_and_split(self):
        official_train, official_test = MVTECADCatalog(self.root, self.categories).scan()
        train, validation = deterministic_normal_split(official_train, 0.2, seed=42)
        return train, validation, official_test

    def test_catalog_and_split_are_deterministic_and_leakage_safe(self) -> None:
        first = self._scan_and_split()
        second = self._scan_and_split()
        self.assertEqual(
            [record.sample_id for record in first[1]],
            [record.sample_id for record in second[1]],
        )
        self.assertEqual((24, 6, 15), tuple(len(split) for split in first))
        assert_protocol_integrity(*first, hash_audit=True)

    def test_test_annotations_are_hidden_outside_evaluation(self) -> None:
        _, _, official_test = self._scan_and_split()
        anomaly = next(record for record in official_test if record.is_anomaly)

        inference_sample = MVTECDataset([anomaly], evaluation=False)[0]
        self.assertNotIn("is_anomaly", inference_sample)
        self.assertNotIn("defect_type", inference_sample)
        self.assertNotIn("mask", inference_sample)

        evaluation_sample = MVTECDataset([anomaly], evaluation=True)[0]
        self.assertTrue(evaluation_sample["is_anomaly"])
        self.assertIsNotNone(evaluation_sample["mask"])

    def test_transform_returns_normalized_chw_image_and_binary_mask(self) -> None:
        _, _, official_test = self._scan_and_split()
        anomaly = next(record for record in official_test if record.is_anomaly)
        transform = MVTecTransform(
            TransformConfig(height=16, width=20, horizontal_flip_probability=0.5, seed=42)
        )
        sample = MVTECDataset([anomaly], transform=transform, evaluation=True)[0]
        self.assertEqual((3, 16, 20), sample["image"].shape)
        self.assertEqual(np.float32, sample["image"].dtype)
        self.assertEqual((1, 16, 20), sample["mask"].shape)
        self.assertEqual({0, 1}, set(np.unique(sample["mask"])))

    def test_missing_ground_truth_mask_fails_closed(self) -> None:
        missing = self.root / "wood" / "ground_truth" / "scratch" / "000_mask.png"
        missing.unlink()
        with self.assertRaises(ProtocolError):
            MVTECADCatalog(self.root, self.categories).scan()

    def test_manifest_is_relative_and_byte_stable(self) -> None:
        train, validation, official_test = self._scan_and_split()
        first = build_manifest(
            self.root,
            train,
            validation,
            official_test,
            seed=42,
            validation_fraction=0.2,
        )
        second = build_manifest(
            self.root,
            train,
            validation,
            official_test,
            seed=42,
            validation_fraction=0.2,
        )
        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )
        first_path = first["splits"]["train"][0]["image_path"]
        self.assertFalse(Path(first_path).is_absolute())
        self.assertNotIn("is_anomaly", first["splits"]["train"][0])
        self.assertNotIn("is_anomaly", first["splits"]["test"][0])
        self.assertNotIn("mask_path", first["splits"]["test"][0])

        evaluation = build_evaluation_manifest(self.root, official_test)
        self.assertEqual("evaluation_only", evaluation["usage"])
        self.assertIn("is_anomaly", evaluation["test_annotations"][0])
        self.assertIn("mask_path", evaluation["test_annotations"][0])

    def test_all_fixture_images_are_decodable(self) -> None:
        train, validation, official_test = self._scan_and_split()
        verify_image_files([*train, *validation, *official_test])

    def test_checksum_inventory_covers_selected_category_files(self) -> None:
        inventory = file_checksum_inventory(self.root, self.categories)
        self.assertEqual("sha256", inventory["algorithm"])
        self.assertEqual(54, inventory["file_count"])
        self.assertEqual(
            sorted(entry["path"] for entry in inventory["files"]),
            [entry["path"] for entry in inventory["files"]],
        )
        self.assertTrue(all(len(entry["sha256"]) == 64 for entry in inventory["files"]))


if __name__ == "__main__":
    unittest.main()

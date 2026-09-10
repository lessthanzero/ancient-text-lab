"""Tests ensuring external data cannot be silently acquired without a pin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ancient_text_lab import ExternalDataset, require_download_pin

ROOT = Path(__file__).resolve().parents[1]


def test_external_manifest_is_explicit_about_manual_access() -> None:
    raw = json.loads((ROOT / "data/manifests/external-datasets.json").read_text())
    datasets = [ExternalDataset.model_validate(row) for row in raw["datasets"]]
    assert {dataset.id for dataset in datasets} == {
        "linear-b-damos",
        "etruscan-openetruscan",
        "linear-b-kober-triplets",
    }
    with pytest.raises(ValueError, match="cannot be fetched automatically"):
        require_download_pin(datasets[0])


def test_download_dataset_requires_url_and_checksum() -> None:
    with pytest.raises(ValueError, match="download datasets require"):
        ExternalDataset.model_validate(
            {
                "id": "bad",
                "source_url": "https://example.org/source",
                "revision": "1",
                "license": "CC BY 4.0",
                "access_mode": "download",
                "notes": "missing pin",
            }
        )

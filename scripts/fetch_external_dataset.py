"""Fetch a checksum-pinned external dataset declared in data/manifests.

Manual-access and unpinned datasets are rejected before any network request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from ancient_text_lab.datasets import ExternalDataset, require_download_pin


def fetch(dataset: ExternalDataset, destination: Path) -> Path:
    url, expected_hash = require_download_pin(dataset)
    with urlopen(str(url), timeout=60) as response:
        payload = response.read()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"checksum mismatch for {dataset.id}: {actual_hash}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id")
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/external-datasets.json")
    )
    args = parser.parse_args()
    raw = json.loads(args.manifest.read_text())
    dataset = next(
        (
            ExternalDataset.model_validate(row)
            for row in raw["datasets"]
            if row["id"] == args.dataset_id
        ),
        None,
    )
    if dataset is None:
        raise SystemExit(f"unknown dataset: {args.dataset_id}")
    print(fetch(dataset, args.destination))


if __name__ == "__main__":
    main()

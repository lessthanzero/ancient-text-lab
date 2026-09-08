"""Contract and fixture validation for the initial public surface."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
from ancient_text_lab import Assertion, EvidenceLayer, EvidenceRecord, NormalizedDocument
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "data" / "schemas" / "v1"
SAMPLE_DIR = ROOT / "data" / "samples"


@pytest.mark.parametrize("sample_name", ["linear-a", "phaistos-disc"])
@pytest.mark.parametrize(
    ("document_name", "schema_name"),
    [
        ("manifest.json", "source-artifact.schema.json"),
        ("document.json", "normalized-document.schema.json"),
        ("evidence.json", "evidence-record.schema.json"),
        ("assertion.json", "assertion.schema.json"),
    ],
)
def test_synthetic_fixture_matches_its_versioned_schema(
    sample_name: str, document_name: str, schema_name: str
) -> None:
    instance = json.loads((SAMPLE_DIR / sample_name / document_name).read_text())
    schema = json.loads((SCHEMA_DIR / schema_name).read_text())
    Draft202012Validator(schema).validate(instance)


@pytest.mark.parametrize("sample_name", ["linear-a", "phaistos-disc"])
def test_normalized_fixture_has_valid_document_invariants(sample_name: str) -> None:
    raw = json.loads((SAMPLE_DIR / sample_name / "document.json").read_text())
    document = NormalizedDocument.model_validate(raw)
    assert all(unit.document_id == document.id for unit in document.units)


@pytest.mark.parametrize("sample_name", ["linear-a", "phaistos-disc"])
def test_fixture_manifest_checksum_matches_its_source_artifact(sample_name: str) -> None:
    fixture_dir = SAMPLE_DIR / sample_name
    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    expected_digest = manifest["artifacts"][0]["sha256"]
    assert sha256((fixture_dir / "source.txt").read_bytes()).hexdigest() == expected_digest


def test_verified_assertion_requires_explicit_promotion_and_evidence() -> None:
    with pytest.raises(ValidationError, match="verification evidence"):
        Assertion.model_validate(
            {
                "id": "bad-verified",
                "kind": "hypothesis",
                "status": "verified",
                "claim": "An untraceable verification.",
                "evidence_ids": ["e1"],
                "confidence": 1.0,
            }
        )


def test_evidence_record_requires_an_epistemic_layer_and_a_source() -> None:
    with pytest.raises(ValidationError):
        EvidenceRecord.model_validate(
            {
                "id": "bad-evidence",
                "layer": EvidenceLayer.OBSERVATION,
                "source_artifact_ids": [],
                "researcher": "researcher",
                "confidence": 1.0,
            }
        )


def test_schema_rejects_a_verified_assertion_without_a_trail() -> None:
    schema = json.loads((SCHEMA_DIR / "assertion.schema.json").read_text())
    invalid = {
        "id": "bad-verified",
        "kind": "hypothesis",
        "status": "verified",
        "claim": "No trace.",
        "evidence_ids": ["e1"],
        "confidence": 1.0,
        "verification_evidence_ids": [],
        "promoted_from_assertion_id": None,
    }
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schema).validate(invalid)

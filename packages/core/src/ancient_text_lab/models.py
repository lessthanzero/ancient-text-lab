"""Versioned, epistemically explicit corpus contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceLayer(str, Enum):
    """The epistemic layer of a record; later layers never overwrite earlier ones."""

    OBSERVATION = "observation"
    TRANSCRIPTION = "transcription"
    INTERPRETATION = "interpretation"
    HYPOTHESIS = "hypothesis"


class AssertionKind(str, Enum):
    TRANSCRIPTION = "transcription"
    INTERPRETATION = "interpretation"
    HYPOTHESIS = "hypothesis"


class AssertionStatus(str, Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    SUPPORTED = "supported"
    FALSIFIED = "falsified"
    VERIFIED = "verified"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceArtifact(ContractModel):
    id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    media_type: str = Field(min_length=1)
    license: str = Field(min_length=1)
    artifact_kind: str = "source"


class Transformation(ContractModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class SignOccurrence(ContractModel):
    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    unit_id: str = Field(min_length=1)
    position: int = Field(ge=0)
    sign_id: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    uncertain: bool = False


class TextUnit(ContractModel):
    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    position: int = Field(ge=0)
    signs: tuple[str, ...] = Field(min_length=1)
    annotations: dict[str, Any] = Field(default_factory=dict)


class NormalizedDocument(ContractModel):
    id: str = Field(min_length=1)
    source_artifact_id: str = Field(min_length=1)
    script: str = Field(min_length=1)
    units: tuple[TextUnit, ...] = Field(min_length=1)
    transformations: tuple[Transformation, ...] = ()

    @model_validator(mode="after")
    def validate_unit_ownership_and_identity(self) -> NormalizedDocument:
        unit_ids = [unit.id for unit in self.units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("document unit IDs must be unique")
        if any(unit.document_id != self.id for unit in self.units):
            raise ValueError("every unit must reference its containing document")
        return self


class EvidenceRecord(ContractModel):
    id: str = Field(min_length=1)
    layer: EvidenceLayer
    source_artifact_ids: tuple[str, ...] = Field(min_length=1)
    transformation: Transformation | None = None
    researcher: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: str | None = None


class Assertion(ContractModel):
    id: str = Field(min_length=1)
    kind: AssertionKind
    status: AssertionStatus
    claim: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    verification_evidence_ids: tuple[str, ...] = ()
    promoted_from_assertion_id: str | None = None

    @model_validator(mode="after")
    def require_explicit_verification_trail(self) -> Assertion:
        if self.status is AssertionStatus.VERIFIED:
            if not self.verification_evidence_ids:
                raise ValueError("verified assertions require verification evidence")
            if not self.promoted_from_assertion_id:
                raise ValueError("verified assertions require an explicit promoted-from assertion")
        return self

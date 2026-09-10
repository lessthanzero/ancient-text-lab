"""External-data registration contracts; downloading never accepts an unpinned artifact."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, HttpUrl, model_validator

from ancient_text_lab.models import ContractModel


class AccessMode(str, Enum):
    DOWNLOAD = "download"
    MANUAL = "manual"


class ExternalDataset(ContractModel):
    id: str = Field(min_length=1)
    source_url: HttpUrl
    revision: str = Field(min_length=1)
    license: str = Field(min_length=1)
    access_mode: AccessMode
    download_url: HttpUrl | None = None
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_a_pinned_download(self) -> ExternalDataset:
        if self.access_mode is AccessMode.DOWNLOAD and (not self.download_url or not self.sha256):
            raise ValueError("download datasets require both download_url and sha256")
        return self


def require_download_pin(dataset: ExternalDataset) -> tuple[HttpUrl, str]:
    """Return a verified download declaration or fail before any network I/O."""
    if dataset.access_mode is not AccessMode.DOWNLOAD:
        raise ValueError(
            f"dataset {dataset.id} requires manual access and cannot be fetched automatically"
        )
    assert dataset.download_url is not None and dataset.sha256 is not None
    return dataset.download_url, dataset.sha256

# Ancient Text Lab

Shared, reproducible research infrastructure for computational analysis of ancient
writing systems. Corpus-specific research remains in independent repositories
(`linear-a`, `phaistos-disk`). Those siblings do **not** yet depend on this package
as an installable import — treat that as future work.

**Status:** `0.2.0` research preview. See [CHANGELOG.md](CHANGELOG.md),
[SCIENTIFIC_LIMITATIONS.md](SCIENTIFIC_LIMITATIONS.md), and [NOTICE](NOTICE).

## Current surface

Core contracts cover provenance/evidence models, deterministic sign-sequence
statistics, and blind-evaluation helpers. Working-tree modules also explore
experimental “frontiers” engines — treat those as demos, not settled science
([docs/FRONTIERS.md](docs/FRONTIERS.md)).

This package does not claim decipherments of Linear A, the Phaistos Disc, or
any other undeciphered script.

## Benchmark pilots

- [Linear A → Linear B](projects/linear-b/README.md) tests recovery of
  citation-backed formal correspondences against known Linear B endpoints. It
  cannot establish a Linear A sound value.
- [Etruscan morphology](projects/etruscan/README.md) tests only source-cited
  segmentation and bounded morphology labels; translation is explicitly out of
  scope.

Both pilots currently contain synthetic fixtures. Their external-source registry
requires a stable revision, licence, and SHA-256 before full-corpus downloads
are allowed.

## Licensing

Software is MIT ([LICENSE](LICENSE)). Third-party / scholarly material is carved
out in [NOTICE](NOTICE) — do not treat external corpora as MIT simply because
they appear in a manifest.

## Development

```bash
uv sync --all-packages --group dev
uv run --package ancient-text-lab pytest
uv run --package ancient-text-lab ruff check packages tests
uv build --package ancient-text-lab
```

See [migration guidance](docs/migration.md) for using a tagged release or a
temporary editable sibling install.

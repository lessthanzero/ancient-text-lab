# Ancient Text Lab

Shared, reproducible research infrastructure for computational analysis of ancient
writing systems. Corpus-specific research remains in independent repositories.

## Current surface

`ancient-text-lab` is deliberately small at first: versioned corpus/evidence
contracts and deterministic sign-sequence statistics. It does not contain
decipherment conclusions, corpus-specific loaders, or linguistic assumptions.

## Development

```bash
uv sync --all-packages --group dev
uv run --package ancient-text-lab pytest
uv run --package ancient-text-lab ruff check packages tests
uv build --package ancient-text-lab
```

See [migration guidance](docs/migration.md) for using a tagged release or a
temporary editable sibling install.


# Ancient Text Lab: decisions for agents

This is the operating record for agents working in this repository. It records
decisions already made; do not reverse them without explicit user direction.

## Purpose and boundaries

- `ancient-text-lab` owns reusable computational-research infrastructure for
  ancient writing systems.
- `../linear-a` and `../phaistos-disk` remain independent research repositories,
  independently runnable and versioned. Do not copy them into this monorepo,
  add submodules, or add permanent sibling-path imports.
- Project-specific hypotheses, corpus quirks, notebooks, visualizers, and
  conclusions stay in their project repositories. Extract only functionality
  with at least two real consumers or clearly generic infrastructure value.
- Do not create empty domain frameworks. The initial public distribution is the
  single `ancient_text_lab` package under `packages/core`.

## Evidence and data policy

- Never turn an interpretation or hypothesis into an observation. Keep
  observation, transcription, interpretation, and hypothesis as separate
  evidence layers.
- `verified` is an assertion status, not an evidence layer. A verified assertion
  must explicitly name both verification evidence and the assertion from which
  it was promoted.
- Corpus data must carry stable IDs, source artifact IDs, a licence, checksum,
  processing/transformation version, confidence, and uncertainty where relevant.
- Keep full external corpora and source images out of Git unless the user
  explicitly changes this policy. Track small synthetic/redistribution-safe
  fixtures, schemas, manifests, and checksums in Git.
- Do not silently fetch unpinned sources. `ExternalDataset` permits automated
  download only when both an exact download URL and SHA-256 are declared.
  DĀMOS and OpenEtruscan are currently manual-access declarations until their
  release-file pins are registered.

## Current public core

- `models.py`: corpus, provenance, evidence, and assertion contracts.
- `sequence.py`: deterministic unit-boundary sequence normalization, transition
  counts, PPMI, and seeded within-unit null shuffles.
- `evaluation.py`: model-visible benchmark cases, separate gold labels,
  predictions, exact-match/coverage reports, provenance checks, and
  group-disjoint split validation.
- `datasets.py`: external dataset registration and pin enforcement.
- Keep clustering, inference policy, palaeography-specific features, and corpus
  loaders outside the core until they are demonstrated across projects.

## Benchmark policy

### Linear A → Linear B

- The first task is recovery of **citation-backed formal sign correspondences**
  against known Linear B sign IDs/values.
- It validates a transfer method; it does **not** establish any Linear A sound
  value or decipherment.
- Real historical cases require family-disjoint splits, citation-backed gold
  labels, source pins, and random-permutation, visual-only, and text-only
  controls before a score may be reported.

### Etruscan

- The first task is held-out segmentation and bounded, source-cited morphology.
- Translation is out of scope. Do not treat a plausible gloss, LLM output, or
  unsourced corpus annotation as gold.
- Real cases must be split by normalized text and lemma/form family, and include
  character-ngram, frequency, and abstention baselines.

### Fixtures and results

- The existing Linear B and Etruscan cases are synthetic contract fixtures only.
  Never describe their results as historical findings.
- Keep public model input cases separate from withheld gold labels. Do not put
  `gold`, `gold_label`, `label`, `target`, or `answer` keys in case features.
- Report abstentions in the denominator; do not discard them to improve a score.

## Research claims: current limits

- Linear A remains undeciphered. Its current contribution is evidence-bounded
  audit infrastructure, including an explicit refusal to infer most damaged
  signs, not a translation or recovered language.
- Phaistos Disc structural repetitions are suitable for description and testing,
  but the repository does not establish a decipherment, hymn, meter, phonetic
  grid, material provenance, or other historical conclusion.
- Treat all previous project claims as hypotheses unless supported by released
  primary data, reproducible methods, appropriate controls, and independent
  scholarly review.

## Development and release

- Use `uv`; run `uv run --package ancient-text-lab ruff check packages tests scripts`,
  `uv run --package ancient-text-lab pytest`, and `uv build --package ancient-text-lab`.
- Do not write core logic in `scripts/`; scripts orchestrate workflows only.
- Build tagged Git releases from this repository. Sibling projects should depend
  on a tagged release, not arbitrary local paths. Editable sibling installs are
  temporary development-only tooling; see `docs/migration.md`.
- Forgejo is the current private source of record at
  a private forge during early development; public GitHub publication is the intended distribution path.

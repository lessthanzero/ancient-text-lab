# Consuming `ancient-text-lab`

Research repositories remain independently versioned and runnable. They should
depend only on a tagged release of this package after its API is stable:

```toml
dependencies = [
  "ancient-text-lab @ git+ssh://git@example.org/research/ancient-text-lab.git@v0.1.0#subdirectory=packages/core",
]
```

During local extraction work only, an editable install is acceptable:

```bash
uv pip install -e ../ancient-text-lab/packages/core
```

Do not commit sibling-relative imports, `PYTHONPATH` changes, submodules, or
copied project data. Establish compatibility with the public contract tests
before replacing a project-local implementation.


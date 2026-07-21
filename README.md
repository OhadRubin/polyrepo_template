# Polyrepo Workspace Library

This project provides a small Python library for publishing a local polyrepo
workspace as one remote workspace state. A workspace is defined by a
`repositories.yaml` manifest that lists each component repository, where it lives
locally, and where it should be reconstructed remotely.

## Core Terms

- **Polyrepo workspace:** A local directory that coordinates multiple independent Git repositories.
- **Repository registry:** The `repositories` mapping in `repositories.yaml`.
- **Launcher:** Application, experiment, test, or deployment code that decides when to publish a workspace.
- **Published workspace:** The immutable result returned by `publish()`, including artifact locations and setup shell.

## Package Layout

```text
polyrepo_template/
|-- pyproject.toml
|-- uv.lock
|-- README.md
`-- src/
    `-- polyrepo/
        |-- __init__.py
        |-- manifest.py
        `-- sync_repo.py
```

## Workspace Manifest

```yaml
workspace:
  name: my-product
  remote_path: my-product-workspace

sync:
  snapshot_path: ~/.polyrepo-snapshots/my-product
  max_file_size_bytes: 10485760
  patch_size_warn_bytes: 1048576

repositories:
  frontend:
    url: git@github.com:my-org/frontend.git
    path: ../frontend

  backend:
    url: git@github.com:my-org/backend.git
    path: ../backend
```

The manifest owns workspace topology, snapshot policy, and repository membership.
The launcher supplies the GCS artifact namespace when it publishes.

## Python API

```python
from pathlib import Path

from polyrepo import sync_repo


workspace_root = Path("~/my-product-workspace").expanduser()
published = sync_repo.publish(workspace_root, "gs://my-product-bucket/polyrepo-sync")
setup = published.render_setup(Path("setup.sh").read_text())
```

`publish()` loads the manifest, builds or reuses a frozen base archive, creates a
patch archive for the current working state, uploads both artifacts, and returns
the shell needed to reconstruct the workspace remotely.

The returned `PublishedWorkspace` provides:

- `manifest_digest`: Identity of the manifest used for publication.
- `gcs_root`: Launcher-selected artifact namespace.
- `base_uri`: GCS location of the frozen base.
- `patch_uri`: GCS location of the current working-state patch.
- `bootstrap_shell`: Commands that reconstruct every registered repository remotely.
- `render_setup(template)`: Injects the reconstruction shell into a launcher setup template.

## Command-Line Sync

Rebuild and upload the frozen base deliberately:

```bash
uv run python -m polyrepo.sync_repo freeze \
  --workspace-root ~/my-product-workspace \
  --gcs-root gs://my-product-bucket/polyrepo-sync
```

Publish the current patch. This creates the base first when the local frozen
snapshot is missing or no longer matches the manifest digest:

```bash
uv run python -m polyrepo.sync_repo publish \
  --workspace-root ~/my-product-workspace \
  --gcs-root gs://my-product-bucket/polyrepo-sync
```

When the package is installed in the active `uv` environment, the same commands
are available through the console entry point:

```bash
uv run polyrepo-sync freeze --workspace-root ~/my-product-workspace --gcs-root gs://my-product-bucket/polyrepo-sync
uv run polyrepo-sync publish --workspace-root ~/my-product-workspace --gcs-root gs://my-product-bucket/polyrepo-sync
```

Both commands print the resolved workspace root, manifest path, manifest digest,
snapshot path, publication GCS root, and manifest-derived local and remote paths
for every repository.

## Synchronization Flow

```text
publish(workspace_root, gcs_root)
  -> read repositories.yaml
  -> construct PolyrepoState
  -> create or reuse the manifest-matched frozen base
  -> compute one patch across the repository registry
  -> upload publication artifacts
  -> generate remote reconstruction shell
  -> return PublishedWorkspace
```

The frozen base is uploaded as `<gcs-root>/base.tar.gz`. Each publication patch
is uploaded as `<gcs-root>/patches/patch_<timestamp>.tar.gz`. Remote
reconstruction extracts the frozen base and overlays the patch.

## Runtime Requirements

- Python 3.10 through 3.12.
- `uv` for local development commands.
- Git repositories present at every local path declared by `repositories.yaml`.
- Authenticated `gsutil` access to the launcher-supplied publication namespace.
- Bash, `gsutil`, `tar`, and `cp` in the remote launcher environment.
- A launcher setup template containing `__SYNC_STR__` when `render_setup()` is used.

## Verification

```bash
uv sync
uv run python -c "from polyrepo import PublishedWorkspace, sync_repo; assert callable(sync_repo.publish); assert PublishedWorkspace"
uv run polyrepo-sync --help
uv build
```

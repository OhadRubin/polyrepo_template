# Polyrepo Workspace Sync

`polyrepo-template` is a small Python package for publishing a group of local
Git repositories as one remote workspace. It is useful when a launcher,
training script, test runner, or deployment job needs to run against the exact
local state of several repositories, including uncommitted files.

The package reads a `polyrepo.yaml` manifest, builds a reusable frozen base
archive, uploads a patch for the current working state, and returns shell code
that reconstructs the workspace on a remote machine.

## What Problem This Solves

Many production jobs are launched from one repository while depending on code in
several sibling repositories. Pushing every work-in-progress change before each
launch is slow and makes experimental branches noisy. Copying files by hand is
fragile.

Polyrepo Workspace Sync gives launchers one narrow contract:

1. Declare the workspace topology in `polyrepo.yaml`.
2. Call `polyrepo.sync_repo.publish(...)` at the launch boundary.
3. Insert the returned setup shell into the remote job bootstrap.

The remote worker receives one workspace directory under `$HOME`, with each
declared repository placed at its manifest path.

## Current Scope

The core `polyrepo` package provides:

- Manifest parsing for a named polyrepo workspace.
- A repository registry keyed by manifest name.
- One canonical frozen-base archive per workspace GCS root.
- Patch archives for tracked and untracked Git files.
- A Python API for launcher code.
- A `polyrepo-sync` CLI for manual freeze and publish operations.
- Setup-shell rendering through the `__SYNC_STR__` placeholder.

Launchers own the target system, credentials, queue, and lifecycle policy.

This distribution also includes:

- `polyrepo_launch` helpers for DAG expansion, execution-mode selection, script
  rendering, and heredoc launches.
- `tpu_dispatch_cli`, which installs the `queue-cli` submission command.

Install the command-line tools from the repository checkout:

```bash
uv tool install .
```

## Planned TPU Queue Integration

The next integration target is a training launcher that renders shell scripts
and enqueues them through a TPU dispatch queue. That launcher will use this
package at the point where it freezes the code that a queued job should run.

The intended flow is:

1. The experiment code builds a task plan.
2. The launcher calls `polyrepo.sync_repo.publish(...)`.
3. The launcher injects the returned sync shell into its setup template.
4. The launcher renders one shell script per task.
5. The launcher uploads each script to GCS.
6. The launcher calls `queue-cli enqueue "gsutil cat ... | bash"` with the
   selected TPU target or affinity query.

This keeps polyrepo synchronization separate from queue policy. The sync package
knows how to publish a workspace. The queue launcher knows how to select TPU
nodes, set worker counts, assign priority, and collect results.

## Repository Layout

```text
polyrepo_template/
|-- README.md
|-- pyproject.toml
|-- uv.lock
`-- src/
    |-- polyrepo/
    |   |-- __init__.py
    |   |-- manifest.py
    |   `-- sync_repo.py
    |-- polyrepo_launch/
    |   |-- __init__.py
    |   |-- dag.py
    |   |-- executor.py
    |   |-- gcs_config.py
    |   |-- generator.py
    |   `-- heredoc_launcher.py
    `-- tpu_dispatch_cli/
        |-- __init__.py
        `-- queue_cli.py
```

## Installation

From a checkout:

```bash
uv sync
```

From another project, install this package from its published Git URL:

```bash
uv add git+https://github.com/<owner>/<repo>.git
```

For local development against a sibling checkout:

```bash
uv add --editable ../polyrepo_template
```

## Requirements

- Python 3.10 through 3.12.
- `uv` for the documented development commands.
- `git`, `rsync`, `tar`, `cp`, and `gsutil` available on the launch machine.
- Authenticated write access to the chosen `gs://...` publication root.
- Remote workers with `bash`, `gsutil`, and `tar`.
- A `polyrepo.yaml` file at the workspace root.

## Manifest

Create `polyrepo.yaml` at the root of the workspace that coordinates the member
repositories and launch storage policy.

```yaml
workspace:
  name: my-product
  remote_path: my-product-workspace

sync:
  snapshot_path: ~/.polyrepo-snapshots/my-product
  max_file_size_bytes: 10485760
  patch_size_warn_bytes: 1048576

launch:
  gcs:
    default_root: gs://example-bucket
    node_ranges:
      - read_bucket: example-read-bucket
        write_bucket: example-write-bucket
        start: 1
        end: 20

repositories:
  frontend:
    url: git@github.com:my-org/frontend.git
    path: ../frontend

  backend:
    url: git@github.com:my-org/backend.git
    path: ../backend
```

### Manifest Fields

`workspace.name` is a stable name for logs and publication paths.

`workspace.remote_path` is the directory created under `$HOME` on the remote
machine. In the example above, the remote workspace root is
`$HOME/my-product-workspace`.

`sync.snapshot_path` is a local cache directory for the frozen base archive and
file hashes. The CLI rebuilds it when `freeze` is run. `publish` reuses it when
the manifest digest still matches.

`sync.max_file_size_bytes` excludes large files from base and patch archives.

`sync.patch_size_warn_bytes` prints a warning when a patch grows large enough
that rebuilding the frozen base may be cleaner.

`launch.gcs.default_root` is the bucket-level GCS URI used by launch helpers
when they need one project-wide storage root. `queue-cli run-script` stores
one-off scripts below `<default-root>/scratch/oneoff`.

`launch.gcs.node_ranges` maps TPU node id ranges to read and write buckets. A
launch target must match exactly one read/write bucket pair.

`repositories.<name>.url` is recorded in publication metadata for launchers and
remote diagnostics.

`repositories.<name>.path` is the local path relative to the workspace root. The
same relative path is used under `workspace.remote_path` on the remote machine.

## Python API

```python
from pathlib import Path

from polyrepo import sync_repo


workspace_root = Path("~/my-product-workspace").expanduser()
published = sync_repo.publish(
    workspace_root,
    "gs://my-product-bucket/polyrepo/my-product",
)

setup_template = Path("setup.sh").read_text()
setup_script = published.render_setup(setup_template)
```

`publish()` returns a `PublishedWorkspace`:

```python
PublishedWorkspace(
    manifest_digest="...",
    gcs_root="gs://my-product-bucket/polyrepo/my-product",
    base_uri="gs://my-product-bucket/polyrepo/my-product/base.tar.gz",
    patch_uri="gs://my-product-bucket/polyrepo/my-product/patches/patch_....tar.gz",
    bootstrap_shell="...",
)
```

`render_setup(template)` replaces `__SYNC_STR__` with the generated bootstrap
shell. The resulting setup script downloads the frozen base, overlays the patch,
and changes directory to the reconstructed workspace.

Example setup template:

```bash
set -euo pipefail

__SYNC_STR__

uv sync
uv run python -m my_launcher.entrypoint
```

## CLI

Rebuild and upload the frozen base:

```bash
uv run polyrepo-sync freeze \
  --workspace-root ~/my-product-workspace \
  --gcs-root gs://my-product-bucket/polyrepo/my-product
```

Publish the current working state:

```bash
uv run polyrepo-sync publish \
  --workspace-root ~/my-product-workspace \
  --gcs-root gs://my-product-bucket/polyrepo/my-product
```

The module entry point is equivalent:

```bash
uv run python -m polyrepo.sync_repo publish \
  --workspace-root ~/my-product-workspace \
  --gcs-root gs://my-product-bucket/polyrepo/my-product
```

Both commands print:

- Workspace root.
- Manifest path.
- Manifest digest.
- Local snapshot path.
- Publication GCS root.
- Local and remote path for each declared repository.

`publish` also prints JSON containing the base URI, patch URI, manifest digest,
and bootstrap shell.

## Sync Model

The base and patches have separate lifecycles:

```text
freeze(workspace_root, gcs_root)
  -> replace the canonical <gcs-root>/base.tar.gz

publish(workspace_root, gcs_root)
  -> load polyrepo.yaml and compute its digest
  -> reuse the canonical <gcs-root>/base.tar.gz
  -> refresh that base when the local frozen cache is missing or stale
  -> compare current files with frozen hashes
  -> upload one patches/patch_<artifact-id>.tar.gz
  -> return remote bootstrap shell
```

The frozen base is uploaded to:

```text
<gcs-root>/base.tar.gz
```

Patches are uploaded to:

```text
<gcs-root>/patches/patch_<artifact-id>.tar.gz
```

The file set comes from:

```bash
git ls-files -co --exclude-standard
```

That includes tracked files and untracked files visible to Git. Keep generated
outputs, datasets, checkpoints, credentials, and other local artifacts behind
Git ignore rules.

## Public-Repo Safety Notes

- Store credentials outside `polyrepo.yaml`, setup templates, and launcher
  scripts.
- Use a private GCS bucket for artifacts that contain unpublished source code.
- Treat patch archives as source-code snapshots.
- Keep large datasets, checkpoints, and generated outputs in object storage or
  Git-ignored paths.
- Review `sync.max_file_size_bytes` before publishing from a workspace that may
  contain local artifacts.

## Launcher Integration Pattern

Launcher code should call `publish()` once per logical launch and pass the
returned setup shell into the remote command template.

```python
from pathlib import Path

from polyrepo import sync_repo


def build_remote_setup(workspace_root: Path, gcs_root: str, template_path: Path) -> str:
    published = sync_repo.publish(workspace_root, gcs_root)
    return published.render_setup(template_path.read_text())
```

For a TPU queue launcher, the rendered task command can then upload a script and
enqueue it:

```bash
queue-cli enqueue "gsutil cat gs://.../run_task.sh | bash" \
  --node-id 21 \
  --n-workers 4 \
  --priority 1 \
  --job-class regular
```

## Development

Run the basic checks:

```bash
uv sync
uv run python -c "from polyrepo import PublishedWorkspace, sync_repo; assert callable(sync_repo.publish); assert PublishedWorkspace"
uv run polyrepo-sync --help
uv run polyrepo-heredoc --help
queue-cli --help
uv build
```

Inspect the CLI:

```bash
uv run polyrepo-sync freeze --help
uv run polyrepo-sync publish --help
```

## Versioning

The package is currently pre-1.0. Public APIs are intentionally small:

- `polyrepo.sync_repo.publish(workspace_root, gcs_root)`
- `polyrepo.sync_repo.freeze(workspace_root, gcs_root)`
- `polyrepo.sync_repo.PublishedWorkspace`
- `polyrepo.manifest.load_state(workspace_root)`

Prefer adding new launcher-specific behavior outside the core `polyrepo` package until it
belongs in the shared synchronization contract.

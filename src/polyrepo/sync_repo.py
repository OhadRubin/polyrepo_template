"""
- description:
    Publish the current working state of every repository declared by a polyrepo
    manifest using the legacy frozen-base plus overlay-patch sync behavior.
- usage:
    uv run python -c "from pathlib import Path; from polyrepo import sync_repo; print(sync_repo.publish(Path('.'), 'gs://my-bucket/polyrepo-sync'))"
    # Call publish from a launcher at its deployment boundary.
    uv run python -m polyrepo.sync_repo freeze --workspace-root . --gcs-root gs://my-bucket/polyrepo-sync
    # Rebuild base.tar.gz from the manifest-driven repository registry.
    uv run python -m polyrepo.sync_repo publish --workspace-root . --gcs-root gs://my-bucket/polyrepo-sync
    # Upload a patch and print publication JSON.
- user_story:
    content:
        As a launcher author, I want one publication call to capture all registered
        repositories so remote work uses the unpublished local workspace state.
    was_generated_via_skill: false
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import posixpath
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .artifact_id import new_artifact_id
from .manifest import MANIFEST_FILENAME, PolyrepoState, RepositoryConfig, load_state


@dataclass(frozen=True, slots=True)
class PublishedWorkspace:
    manifest_digest: str
    gcs_root: str
    base_uri: str
    patch_uri: str
    bootstrap_shell: str

    def render_setup(self, template: str) -> str:
        assert "__SYNC_STR__" in template, "setup template must contain __SYNC_STR__"
        return template.replace("__SYNC_STR__", self.bootstrap_shell)


@dataclass(frozen=True, slots=True)
class GsutilArtifactStore:
    executable: str

    def upload(self, source: Path, destination: str) -> None:
        subprocess.run(
            [self.executable, "cp", str(source), destination],
            check=True,
        )


@dataclass(frozen=True, slots=True)
class PublicationLayout:
    """Give the frozen base a stable URI and each patch a publication URI."""

    gcs_root: str

    @property
    def frozen_base_uri(self) -> str:
        return f"{self.gcs_root}/base.tar.gz"

    def patch_uri(self, patch_id: str) -> str:
        return f"{self.gcs_root}/patches/patch_{patch_id}.tar.gz"


def _normalize_gcs_root(gcs_root: str) -> str:
    normalized = gcs_root.strip().rstrip("/")
    assert normalized.startswith("gs://") and len(normalized) > len("gs://"), gcs_root
    return normalized


@contextlib.contextmanager
def _exclusive_snapshot_lock(snapshot_path: Path) -> Iterator[None]:
    assert snapshot_path.is_absolute(), snapshot_path
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = snapshot_path.with_name(f".{snapshot_path.name}.lock")
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _remote_home_path(state: PolyrepoState, repository: RepositoryConfig) -> str:
    workspace_path = state.workspace.remote_path.as_posix()
    declared_path = repository.declared_path.as_posix()
    assert not posixpath.isabs(workspace_path), workspace_path
    assert not posixpath.isabs(declared_path), declared_path
    target = posixpath.normpath(posixpath.join(workspace_path, declared_path))
    assert target not in ("", ".", ".."), target
    assert not target.startswith("../"), target
    return target


def _workspace_home_path(state: PolyrepoState) -> str:
    workspace_path = posixpath.normpath(state.workspace.remote_path.as_posix())
    assert workspace_path not in ("", ".", ".."), workspace_path
    assert not workspace_path.startswith(("/", "../")), workspace_path
    return workspace_path


def _git_ls_files(repository: RepositoryConfig) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        capture_output=True,
        text=True,
        cwd=repository.local_path,
        check=True,
    )
    return [path for path in result.stdout.strip().split("\n") if path]


def _filter_big_files(
    repository: RepositoryConfig,
    files: list[str],
    max_size_bytes: int,
) -> list[str]:
    kept: list[str] = []
    missing_count = 0
    broken_symlink_count = 0
    oversized_count = 0
    for relative_path in files:
        file_path = repository.local_path / relative_path
        if not file_path.exists():
            if file_path.is_symlink():
                broken_symlink_count += 1
            else:
                missing_count += 1
            continue
        if not file_path.is_file():
            kept.append(relative_path)
            continue
        size = file_path.stat().st_size
        if size > max_size_bytes:
            oversized_count += 1
            continue
        kept.append(relative_path)
    if missing_count:
        print(f"  {repository.name}: skipped {missing_count} missing file(s)")
    if broken_symlink_count:
        print(f"  {repository.name}: skipped {broken_symlink_count} broken symlink(s)")
    if oversized_count:
        print(
            f"  {repository.name}: skipped {oversized_count} file(s) larger than "
            f"{max_size_bytes / 1024 / 1024:.0f} MB"
        )
    return kept


def _compute_file_hash(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rsync_copy(file_list: list[str], source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as file_handle:
        file_handle.write("\n".join(file_list))
        filelist_path = file_handle.name
    try:
        subprocess.run(
            [
                "rsync",
                "-aL",
                "--ignore-missing-args",
                f"--files-from={filelist_path}",
                f"{source}/",
                f"{destination}/",
            ],
            check=True,
        )
    finally:
        Path(filelist_path).unlink()


def _repository_snapshot_path(
    state: PolyrepoState,
    repository: RepositoryConfig,
) -> Path:
    return state.sync.snapshot_path / _remote_home_path(state, repository)


def _hashes_path(state: PolyrepoState, repository: RepositoryConfig) -> Path:
    return _repository_snapshot_path(state, repository) / ".hashes.json"


def _archive_members(state: PolyrepoState) -> list[str]:
    members = {
        _workspace_home_path(state),
        *(_remote_home_path(state, repository) for repository in state.repositories.values()),
    }
    return sorted(members)


def _create_base_archive(state: PolyrepoState) -> None:
    base_archive = state.sync.snapshot_path / "base.tar.gz"
    if base_archive.exists():
        base_archive.unlink()
    subprocess.run(
        [
            "tar",
            "czf",
            str(base_archive),
            "-C",
            str(state.sync.snapshot_path),
            *_archive_members(state),
        ],
        check=True,
    )


def _freeze(
    state: PolyrepoState,
    artifact_store: GsutilArtifactStore,
    publication_layout: PublicationLayout,
) -> None:
    if state.sync.snapshot_path.exists():
        shutil.rmtree(state.sync.snapshot_path)
    state.sync.snapshot_path.mkdir(parents=True)

    workspace_path = state.sync.snapshot_path / _workspace_home_path(state)
    workspace_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(state.manifest_path, workspace_path / MANIFEST_FILENAME)

    for repository in state.repositories.values():
        assert repository.local_path.exists(), repository.local_path
        files = _filter_big_files(
            repository,
            _git_ls_files(repository),
            state.sync.max_file_size_bytes,
        )
        if not files:
            raise RuntimeError(f"No files found in {repository.local_path}")

        frozen_repository = _repository_snapshot_path(state, repository)
        _rsync_copy(files, repository.local_path, frozen_repository)

        hashes: dict[str, str] = {}
        for relative_path in files:
            file_path = frozen_repository / relative_path
            if file_path.is_file():
                hashes[relative_path] = _compute_file_hash(file_path)
        _hashes_path(state, repository).write_text(json.dumps(hashes), encoding="utf-8")
        print(f"  frozen {repository.name} ({len(files)} files, {len(hashes)} hashed)")

    (state.sync.snapshot_path / "freeze_meta.json").write_text(
        json.dumps(
            {
                "frozen_at": datetime.now().isoformat(),
                "manifest_digest": state.manifest_digest,
            }
        ),
        encoding="utf-8",
    )
    _create_base_archive(state)
    artifact_store.upload(
        state.sync.snapshot_path / "base.tar.gz",
        publication_layout.frozen_base_uri,
    )
    print(f"Frozen snapshot uploaded to {publication_layout.frozen_base_uri}")


def _frozen_base_matches(state: PolyrepoState) -> bool:
    meta_path = state.sync.snapshot_path / "freeze_meta.json"
    if not meta_path.exists():
        return False
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("manifest_digest") != state.manifest_digest:
        return False
    if not (state.sync.snapshot_path / "base.tar.gz").is_file():
        return False
    return all(
        _hashes_path(state, repository).is_file()
        for repository in state.repositories.values()
    )


def _print_state_summary(state: PolyrepoState, publication_gcs_root: str) -> None:
    print(f"workspace_root={state.workspace_root}")
    print(f"manifest={state.manifest_path}")
    print(f"manifest_digest={state.manifest_digest}")
    print(f"snapshot_path={state.sync.snapshot_path}")
    print(f"gcs_root={publication_gcs_root}")
    print("repositories:")
    for repository in state.repositories.values():
        print(
            f"  {repository.name}: local={repository.local_path} "
            f"remote=$HOME/{_remote_home_path(state, repository)}"
        )


def _changed_files(state: PolyrepoState, repository: RepositoryConfig) -> list[str]:
    frozen_hashes = json.loads(_hashes_path(state, repository).read_text(encoding="utf-8"))
    files = _filter_big_files(
        repository,
        _git_ls_files(repository),
        state.sync.max_file_size_bytes,
    )

    changed_files: list[str] = []
    for relative_path in files:
        file_path = repository.local_path / relative_path
        if not file_path.is_file():
            continue
        current_hash = _compute_file_hash(file_path)
        if relative_path not in frozen_hashes or frozen_hashes[relative_path] != current_hash:
            changed_files.append(relative_path)
    return changed_files


def _copy_patch_file(repository: RepositoryConfig, relative_path: str, destination: Path) -> None:
    source = repository.local_path / relative_path
    target = destination / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-L", str(source), str(target)], check=True)


def _compute_patch(
    state: PolyrepoState,
    artifact_store: GsutilArtifactStore,
    publication_layout: PublicationLayout,
) -> str:
    patch_id = new_artifact_id()
    patch_directory = Path(tempfile.gettempdir()) / f"patch_{patch_id}"
    patch_directory.mkdir()

    try:
        for repository in state.repositories.values():
            changed_files = _changed_files(state, repository)
            repository_patch_directory = patch_directory / _remote_home_path(state, repository)
            for relative_path in changed_files:
                _copy_patch_file(repository, relative_path, repository_patch_directory)
            print(f"  {repository.name}: {len(changed_files)} changed files")

        patch_tar_name = f"patch_{patch_id}.tar.gz"
        patch_tar_path = Path(tempfile.gettempdir()) / patch_tar_name
        subprocess.run(
            ["tar", "czf", str(patch_tar_path), "-C", str(patch_directory), "."],
            check=True,
        )

        patch_size = patch_tar_path.stat().st_size
        if patch_size > state.sync.patch_size_warn_bytes:
            print(
                f"  WARNING: patch is {patch_size / 1024 / 1024:.1f} MB "
                f"(>{state.sync.patch_size_warn_bytes / 1024 / 1024:.0f} MB). "
                "Consider re-freezing."
            )

        patch_uri = publication_layout.patch_uri(patch_id)
        artifact_store.upload(patch_tar_path, patch_uri)
        patch_tar_path.unlink()
        return patch_uri
    finally:
        shutil.rmtree(patch_directory)


def _bootstrap_shell(state: PolyrepoState, base_uri: str, patch_uri: str) -> str:
    workspace_path = _workspace_home_path(state)
    patch_tar_name = posixpath.basename(patch_uri)
    return f'''echo "=== Syncing polyrepo workspace from base+patch ==="
POLYREPO_WORKSPACE_ROOT="$HOME"/{shlex.quote(workspace_path)}
gsutil cp "{base_uri}" /tmp/base.tar.gz
tar xzf /tmp/base.tar.gz -C "$HOME"
rm /tmp/base.tar.gz
gsutil cp "{patch_uri}" /tmp/{patch_tar_name}
tar xzf /tmp/{patch_tar_name} -C "$HOME"
rm /tmp/{patch_tar_name}
cd "$POLYREPO_WORKSPACE_ROOT"
'''


def publish(workspace_root: Path, gcs_root: str) -> PublishedWorkspace:
    state = load_state(workspace_root)
    publication_layout = PublicationLayout(_normalize_gcs_root(gcs_root))
    artifact_store = GsutilArtifactStore(executable="gsutil")

    with _exclusive_snapshot_lock(state.sync.snapshot_path):
        if not _frozen_base_matches(state):
            print("Frozen base is missing or manifest-mismatched; rebuilding.")
            _freeze(state, artifact_store, publication_layout)
        else:
            print(f"Using existing frozen base at {state.sync.snapshot_path / 'base.tar.gz'}")
        patch_uri = _compute_patch(state, artifact_store, publication_layout)

    return PublishedWorkspace(
        manifest_digest=state.manifest_digest,
        gcs_root=publication_layout.gcs_root,
        base_uri=publication_layout.frozen_base_uri,
        patch_uri=patch_uri,
        bootstrap_shell=_bootstrap_shell(
            state,
            publication_layout.frozen_base_uri,
            patch_uri,
        ),
    )


def freeze(workspace_root: Path, gcs_root: str) -> str:
    state = load_state(workspace_root)
    publication_layout = PublicationLayout(_normalize_gcs_root(gcs_root))
    artifact_store = GsutilArtifactStore(executable="gsutil")
    with _exclusive_snapshot_lock(state.sync.snapshot_path):
        _freeze(state, artifact_store, publication_layout)
    return publication_layout.frozen_base_uri


def _parse_args(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="polyrepo-sync",
        description="Manifest-driven frozen-base and patch publication.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser(
        "freeze",
        help="Rebuild and upload base.tar.gz from polyrepo.yaml.",
    )
    freeze_parser.add_argument("--workspace-root", type=Path, required=True)
    freeze_parser.add_argument("--gcs-root", required=True)

    publish_parser = subparsers.add_parser(
        "publish",
        help="Ensure a manifest-matched base exists, upload a patch, and print publication JSON.",
    )
    publish_parser.add_argument("--workspace-root", type=Path, required=True)
    publish_parser.add_argument("--gcs-root", required=True)

    return parser.parse_args(arguments)


def _print_publication(publication: PublishedWorkspace) -> None:
    print(
        json.dumps(
            {
                "manifest_digest": publication.manifest_digest,
                "gcs_root": publication.gcs_root,
                "base_uri": publication.base_uri,
                "patch_uri": publication.patch_uri,
                "bootstrap_shell": publication.bootstrap_shell,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _cli_freeze(args: argparse.Namespace) -> None:
    state = load_state(args.workspace_root)
    publication_gcs_root = _normalize_gcs_root(args.gcs_root)
    print("=== polyrepo-sync freeze ===")
    _print_state_summary(state, publication_gcs_root)
    base_uri = freeze(args.workspace_root, args.gcs_root)
    print("Freeze complete.")
    print(json.dumps({"base_uri": base_uri}, indent=2, sort_keys=True))


def _cli_publish(args: argparse.Namespace) -> None:
    state = load_state(args.workspace_root)
    publication_gcs_root = _normalize_gcs_root(args.gcs_root)
    print("=== polyrepo-sync publish ===")
    _print_state_summary(state, publication_gcs_root)
    _print_publication(publish(args.workspace_root, args.gcs_root))


def main() -> None:
    args = _parse_args(sys.argv[1:])
    commands = {
        "freeze": _cli_freeze,
        "publish": _cli_publish,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

"""
- description:
    Load repositories.yaml into immutable nested configuration objects and a
    name-addressable repository registry for synchronization operations.
- usage:
    uv run python -c "from pathlib import Path; from polyrepo.manifest import load_state; print(load_state(Path('.')))"
    # Point load_state at a workspace containing repositories.yaml.
- user_story:
    content:
        As a synchronization implementation, I want the workspace manifest converted
        into immutable state so every operation uses the same workspace topology
        and repository registry.
    was_generated_via_skill: false
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import yaml


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    name: str
    remote_path: PurePosixPath


@dataclass(frozen=True, slots=True)
class SyncConfig:
    snapshot_path: Path
    max_file_size_bytes: int
    patch_size_warn_bytes: int


@dataclass(frozen=True, slots=True)
class RepositoryConfig:
    name: str
    url: str
    declared_path: PurePosixPath
    local_path: Path


@dataclass(frozen=True, slots=True)
class RepositoryRegistry(Mapping[str, RepositoryConfig]):
    _entries: Mapping[str, RepositoryConfig]

    def __getitem__(self, name: str) -> RepositoryConfig:
        return self._entries[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True, slots=True)
class PolyrepoState:
    workspace_root: Path
    manifest_path: Path
    manifest_digest: str
    workspace: WorkspaceConfig
    sync: SyncConfig
    repositories: RepositoryRegistry


def load_state(workspace_root: Path) -> PolyrepoState:
    resolved_workspace_root = workspace_root.expanduser().resolve()
    manifest_path = resolved_workspace_root / "repositories.yaml"
    manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    canonical_manifest = json.dumps(
        manifest_data,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest_digest = hashlib.sha256(canonical_manifest).hexdigest()

    workspace_data = manifest_data["workspace"]
    sync_data = manifest_data["sync"]
    repository_data = manifest_data["repositories"]
    assert repository_data, "repositories.yaml must declare at least one repository"

    snapshot_path = Path(sync_data["snapshot_path"]).expanduser()
    if not snapshot_path.is_absolute():
        snapshot_path = (resolved_workspace_root / snapshot_path).resolve()

    entries = {
        name: RepositoryConfig(
            name=name,
            url=values["url"],
            declared_path=PurePosixPath(values["path"]),
            local_path=(resolved_workspace_root / values["path"]).resolve(),
        )
        for name, values in repository_data.items()
    }

    return PolyrepoState(
        workspace_root=resolved_workspace_root,
        manifest_path=manifest_path,
        manifest_digest=manifest_digest,
        workspace=WorkspaceConfig(
            name=workspace_data["name"],
            remote_path=PurePosixPath(workspace_data["remote_path"]),
        ),
        sync=SyncConfig(
            snapshot_path=snapshot_path,
            max_file_size_bytes=sync_data["max_file_size_bytes"],
            patch_size_warn_bytes=sync_data["patch_size_warn_bytes"],
        ),
        repositories=RepositoryRegistry(MappingProxyType(entries)),
    )

"""
- description:
    Resolve bucket-level GCS roots and node-range bucket mappings from the
    launch section of polyrepo.yaml.
- usage:
    uv run python -c "from polyrepo_launch.gcs_config import resolve_default_gcs_root; print(resolve_default_gcs_root())"
    # Run from the workspace directory that contains polyrepo.yaml.
- user_story:
    content:
        As a launcher author, Ohad wants queue launch code to resolve default
        and per-node GCS roots from the same manifest that defines the workspace
        repositories.
    was_generated_via_skill: false
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from polyrepo.manifest import MANIFEST_FILENAME, LaunchGcsConfig, load_state


@dataclass(frozen=True)
class GcsRoots:
    read_root: str
    write_root: str


def workspace_root() -> Path:
    root = Path.cwd().resolve()
    assert (root / MANIFEST_FILENAME).is_file(), root
    return root


def resolve_gcs_root(value: str) -> str:
    root = value.strip().rstrip("/")
    bucket = root.removeprefix("gs://")
    assert root.startswith("gs://") and bucket and "/" not in bucket, value
    return root


def launch_gcs_config() -> LaunchGcsConfig:
    return load_state(workspace_root()).launch.gcs


def resolve_default_gcs_root() -> str:
    return resolve_gcs_root(launch_gcs_config().default_root)


def resolve_gcs_roots_for_node_ids(node_ids: tuple[int, ...]) -> GcsRoots:
    gcs = launch_gcs_config()
    # This resolver owns bucket policy selection; dispatch owns fleet node validity.
    policies = {
        (node_range.read_bucket, node_range.write_bucket)
        for node_id in node_ids
        for node_range in gcs.node_ranges
        if node_range.start <= node_id <= node_range.end
    }
    assert len(policies) == 1, (node_ids, policies)
    read_bucket, write_bucket = next(iter(policies))
    return GcsRoots(
        read_root=resolve_gcs_root(f"gs://{read_bucket}"),
        write_root=resolve_gcs_root(f"gs://{write_bucket}"),
    )

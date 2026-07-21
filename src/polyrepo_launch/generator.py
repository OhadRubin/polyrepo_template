"""Stable queueing machinery for training-script render callbacks."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import shlex
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from jinja2 import Environment, StrictUndefined, meta

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"google\.api_core\._python_version_support",
)

from cloudpathlib import CloudPath
from dotenv import load_dotenv
from polyrepo import sync_repo as polyrepo_sync
from polyrepo.manifest import load_state as load_polyrepo_state

from .gcs_config import GcsRoots, resolve_gcs_roots_for_node_ids, workspace_root


JINJA_ENV = Environment(undefined=StrictUndefined)
OUTPUT_ROOT = "generated_training_scripts"
WANDB_KEY_ENV_VAR = "WANDB_API_KEY"
TPU_NODE_RE = re.compile(r"^(?P<tpu_type>[^-]+)-(?P<n_chips>\d+)-node-(?P<node_id>\d+)$")
WORKERS_PER_TPU_SHAPE = {
    ("v4", 256): 32,
    ("v5p", 32): 4,
    ("v6e", 16): 4,
}
NODE_ID_RANGES_PER_TPU_SHAPE = {
    ("v4", 256): range(41, 61),
    ("v5p", 32): range(21, 41),
    ("v6e", 16): range(1, 21),
}
V5P_WORKERS_PER_NODE = WORKERS_PER_TPU_SHAPE[("v5p", 32)]
V5P_NODE_ID_MIN = 21
V5P_NODE_ID_MAX = 40


@dataclass(frozen=True)
class RenderedScript:
    filename: str
    content: str


@dataclass(frozen=True)
class RenderSpec:
    filename: str
    fragments: tuple[str, ...]
    variables: dict[str, object]


@dataclass(frozen=True)
class LaunchOptions:
    launch_node_ids: tuple[int, ...]
    launch_affinity_query: str | None
    gcs_roots: GcsRoots
    workers_per_node: int
    queue_priority: int
    job_class: str
    dry_run: bool
    output_dir: Path


@dataclass(frozen=True)
class RepositoryIdentity:
    commit: str
    dirty: bool


class TaskRenderer(Protocol):
    def __call__(self, task: Any, setup: str, gcs_roots: GcsRoots) -> RenderedScript:
        raise NotImplementedError


def render_blocks(fragments: tuple[str, ...], variables: dict[str, object]) -> str:
    script = "\n".join(fragments)
    undeclared_variables = meta.find_undeclared_variables(JINJA_ENV.parse(script))
    inputs = set(variables)
    missing = undeclared_variables - inputs
    unused = inputs - undeclared_variables
    assert not missing and not unused, (
        f"Template input mismatch: missing={sorted(missing)}, unused={sorted(unused)}"
    )
    return JINJA_ENV.from_string(script).render(**variables)


def render_script(spec: RenderSpec) -> RenderedScript:
    assert spec.filename
    assert spec.fragments
    return RenderedScript(
        filename=spec.filename,
        content=render_blocks(spec.fragments, spec.variables),
    )


def _load_wandb_key() -> str:
    root = workspace_root()
    load_dotenv(root / ".env")
    key = os.environ.get(WANDB_KEY_ENV_VAR, "").strip()
    if not key:
        key_path = root / ".wandbkey"
        if key_path.exists():
            key = key_path.read_text().strip()
    if key:
        return key
    raise RuntimeError(
        f"Missing W&B API key. Add {WANDB_KEY_ENV_VAR}=... to .env "
        "or put the key in .wandbkey."
    )


def _dry_run_workspace_target_shell() -> str:
    state = load_polyrepo_state(workspace_root())
    workspace_path = posixpath.normpath(state.workspace.remote_path.as_posix())
    assert workspace_path not in ("", ".", ".."), workspace_path
    assert not workspace_path.startswith(("/", "../")), workspace_path
    lines = [
        "set -euo pipefail",
        f'POLYREPO_WORKSPACE_ROOT="$HOME"/{shlex.quote(workspace_path)}',
    ]
    for repository in state.repositories.values():
        target = posixpath.normpath(
            posixpath.join(workspace_path, repository.declared_path.as_posix())
        )
        assert target not in ("", ".", ".."), target
        assert not target.startswith(("/", "../")), target
        lines.append(f'POLYREPO_TARGET="$HOME"/{shlex.quote(target)}')
    lines.append('cd "$POLYREPO_WORKSPACE_ROOT"')
    return "\n".join(lines) + "\n"


def _repository_identity(repository: Path) -> RepositoryIdentity:
    assert (repository / ".git").exists(), repository
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert commit
    return RepositoryIdentity(commit=commit, dirty=bool(status))


def _parse_launch_nodes(raw: list[str] | None) -> tuple[int, ...]:
    if raw is None:
        return ()
    node_ids = []
    for raw_part in raw:
        for part in raw_part.split(","):
            part = part.strip()
            if not part:
                continue
            node_ids.extend(_parse_launch_node_part(part))
    if not node_ids:
        raise ValueError("--launch must contain at least one node id")
    return tuple(node_ids)


def _parse_launch_node_part(part: str) -> tuple[int, ...]:
    match = TPU_NODE_RE.match(part)
    if match is not None:
        tpu_shape = (match.group("tpu_type"), int(match.group("n_chips")))
        assert tpu_shape in NODE_ID_RANGES_PER_TPU_SHAPE, tpu_shape
        node_id = int(match.group("node_id"))
        assert node_id in NODE_ID_RANGES_PER_TPU_SHAPE[tpu_shape], node_id
        return (node_id,)
    if "-" in part:
        start, end = part.split("-", 1)
        assert int(start) <= int(end), part
        return tuple(range(int(start), int(end) + 1))
    return (int(part),)


def _node_affinity_query(node_ids: tuple[int, ...]) -> str:
    assert node_ids, node_ids
    sorted_node_ids = sorted(set(node_ids))
    if len(sorted_node_ids) == 1:
        return f"node_id = {sorted_node_ids[0]}"
    return "node_id IN (" + ", ".join(str(node_id) for node_id in sorted_node_ids) + ")"


def infer_workers_per_node(raw: list[str] | None) -> int | None:
    if raw is None:
        return None
    worker_counts = set()
    for node_id in _parse_launch_nodes(raw):
        matching_worker_counts = {
            workers
            for tpu_shape, node_ids in NODE_ID_RANGES_PER_TPU_SHAPE.items()
            for workers in (WORKERS_PER_TPU_SHAPE[tpu_shape],)
            if node_id in node_ids
        }
        assert matching_worker_counts, node_id
        worker_counts.update(matching_worker_counts)
    assert len(worker_counts) == 1, worker_counts
    return next(iter(worker_counts))


def _parse_args() -> LaunchOptions:
    parser = argparse.ArgumentParser(
        description="Render and enqueue the training tasks defined by the caller."
    )
    launch_target = parser.add_mutually_exclusive_group()
    launch_target.add_argument(
        "--launch",
        nargs="+",
        help=(
            "Node ids or TPU names. Supports comma-separated node ids and shell "
            "expansion like v5p-32-node-{21..40}. Required unless --dry-run is used."
        ),
    )
    launch_target.add_argument(
        "--launch-affinity",
        nargs="+",
        help=(
            "Node ids or TPU names that any queued task may run on. Supports "
            "comma-separated ids, ranges like 1-20, and shell expansion."
        ),
    )
    parser.add_argument("--workers-per-node", "--n-workers", type=int)
    parser.add_argument("--queue-priority", "--priority", type=int, default=1)
    parser.add_argument("--job-class", choices=("regular", "idle"), default="regular")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render scripts locally without freezing, uploading, or enqueueing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "polyrepo-launch-preview",
        help="Local destination used by --dry-run.",
    )
    args = parser.parse_args()
    launch_node_ids = _parse_launch_nodes(args.launch)
    launch_affinity_node_ids = _parse_launch_nodes(args.launch_affinity)
    if not (launch_node_ids or launch_affinity_node_ids):
        parser.error("--launch or --launch-affinity is required")
    workers_per_node = args.workers_per_node
    if workers_per_node is None:
        launch_raw = args.launch or args.launch_affinity
        workers_per_node = infer_workers_per_node(launch_raw) or V5P_WORKERS_PER_NODE
    return LaunchOptions(
        launch_node_ids=launch_node_ids,
        launch_affinity_query=(
            _node_affinity_query(launch_affinity_node_ids)
            if launch_affinity_node_ids else None
        ),
        gcs_roots=resolve_gcs_roots_for_node_ids(launch_node_ids or launch_affinity_node_ids),
        workers_per_node=workers_per_node,
        queue_priority=args.queue_priority,
        job_class=args.job_class,
        dry_run=args.dry_run,
        output_dir=args.output_dir.resolve(),
    )


def queue_command(cloud_path: object, task_idx: int, options: LaunchOptions) -> list[str]:
    queue_cmd = [
        "queue-cli",
        "enqueue",
        f"gsutil cat {cloud_path} | bash",
        "--n-workers",
        str(options.workers_per_node),
        "--priority",
        str(options.queue_priority),
        "--job-class",
        options.job_class,
    ]
    if options.launch_affinity_query is not None:
        queue_cmd.extend(["--affinity", options.launch_affinity_query])
        return queue_cmd
    node_id = options.launch_node_ids[task_idx % len(options.launch_node_ids)]
    queue_cmd.extend(["--node-id", str(node_id)])
    return queue_cmd


def execute_tasks(
    tasks: tuple[Any, ...],
    render_task: TaskRenderer,
    options: LaunchOptions,
) -> None:
    print(f"=== Generated {len(tasks)} task(s) ===")
    root = workspace_root()
    generated_script_run_id = (
        "dry_run" if options.dry_run else datetime.now().strftime("%H%M%S_%Y%m%d")
    )
    setup_template = (root / "setup.sh").read_text()
    polyrepo_state = load_polyrepo_state(root)
    workspace_name = polyrepo_state.workspace.name
    assert workspace_name and "/" not in workspace_name, workspace_name
    polyrepo_publication_root = f"{options.gcs_roots.write_root}/polyrepo/{workspace_name}"
    repository_identities = {
        name: {
            "commit": identity.commit,
            "dirty": identity.dirty,
            "path": repository.declared_path.as_posix(),
            "url": repository.url,
        }
        for name, repository in polyrepo_state.repositories.items()
        for identity in (_repository_identity(repository.local_path),)
    }
    repositories_json = json.dumps(
        repository_identities,
        sort_keys=True,
        separators=(",", ":"),
    )
    if options.dry_run:
        setup = setup_template.replace(
            "__SYNC_STR__", _dry_run_workspace_target_shell()
        )
        polyrepo_manifest_digest = polyrepo_state.manifest_digest
        polyrepo_base_uri = f"{polyrepo_publication_root}/base.tar.gz"
        polyrepo_patch_uri = "DRY_RUN_UNPUBLISHED"
        wandb_key = "DRY_RUN_WANDB_KEY"
    else:
        published = polyrepo_sync.publish(root, polyrepo_publication_root)
        setup = published.render_setup(setup_template)
        polyrepo_manifest_digest = published.manifest_digest
        polyrepo_base_uri = published.base_uri
        polyrepo_patch_uri = published.patch_uri
        wandb_key = _load_wandb_key()
    setup = (
        setup.replace("__WANDB_KEY__", wandb_key)
        .replace("__READ_GCS_ROOT__", options.gcs_roots.read_root)
        .replace(
            "__POLYREPO_REPOSITORIES_JSON__",
            shlex.quote(repositories_json),
        )
        .replace("__POLYREPO_MANIFEST_DIGEST__", polyrepo_manifest_digest)
        .replace("__POLYREPO_BASE_URI__", polyrepo_base_uri)
        .replace("__POLYREPO_PATCH_URI__", polyrepo_patch_uri)
    )
    setup = f"{setup}\nexport GENERATED_SCRIPT_RUN_ID={shlex.quote(generated_script_run_id)}\n"

    if options.dry_run:
        options.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        cloud_root = CloudPath(options.gcs_roots.write_root) / "runs" / OUTPUT_ROOT / generated_script_run_id

    for task_idx, task in enumerate(tasks):
        rendered = render_task(task, setup, options.gcs_roots)
        if options.dry_run:
            destination = options.output_dir / rendered.filename
            destination.write_text(rendered.content)
            print(f"  rendered: {destination}")
            continue

        cloud_path = cloud_root / rendered.filename
        with cloud_path.open("w+") as output:
            output.write(rendered.content)

        queue_cmd = queue_command(cloud_path, task_idx, options)
        print(" ".join(queue_cmd))
        subprocess.run(queue_cmd, check=True)


def main(tasks: tuple[Any, ...], render_task: TaskRenderer) -> None:
    execute_tasks(tasks, render_task, _parse_args())

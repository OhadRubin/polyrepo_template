#!/usr/bin/env python3
"""
- description:
    Launch a repo-local bash payload through any generator-shaped script that
    delegates to polyrepo_launch.executor.run(...), then block until
    every worker uploads its combined setup+payload log and exit code to GCS.
    The payload travels with the polyrepo sync, so each worker runs it via
    `bash` from the synced workspace; the worker rank comes from the
    `agent-worker-number` instance-metadata attribute.
- usage:
    # Run scratch.sh on every worker of node 21, then print the last 50 log lines per worker.
    uv run polyrepo-heredoc --launch v5p-32-node-21 --heredoc-file scratch.sh --lines 50

    # Use a different generator-shaped script while keeping the same queue harness.
    uv run polyrepo-heredoc --generator-script scripts/my_plan.py --launch v6e-16-node-2 --heredoc-file scratch.sh --lines 80

    # Pass the payload as an actual heredoc on stdin.
    uv run polyrepo-heredoc --launch-affinity v6e-16-node-{1..20} --workers-per-node 4 --lines 50 <<'EOF'
    uv run python -c "import jax; print(jax.devices())"
    EOF

    # Wait for all workers, but print only rank 0's log tail.
    uv run polyrepo-heredoc --launch v5p-32-node-21 --lines 50 --print-only-worker 0 <<'EOF'
    uv run python -c "import jax; print(jax.devices())"
    EOF
- user_story:
    content:
        Ohad wants to run short one-off scripts on the real production TPU
        environment while choosing the experiment generator at launch time. He
        points the launcher at any repo-local generator-shaped script and gets
        back per-worker logs under
        <gcs>/runs/heredoc_results/<generator>_<payload>_<ts>/{rank}.txt.
    was_generated_via_skill: false
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from cloudpathlib import CloudPath
from polyrepo.artifact_id import new_artifact_id

from .gcs_config import resolve_gcs_roots_for_node_ids, workspace_root
from .generator import _parse_launch_nodes, infer_workers_per_node


POLL_SECONDS = 15
DEFAULT_GENERATOR_SCRIPT = Path("scripts") / "make_training_scripts.py"


def repo_local_file(path: Path, root: Path) -> Path:
    local_path = path if path.is_absolute() else root / path
    local_path = local_path.resolve()
    assert local_path.is_file(), local_path
    local_path.relative_to(root)
    return local_path


def generator_command(
    generator_script: Path,
    launch_flag: str,
    launch_targets: list[str],
    heredoc_file: Path,
    result_root: str,
    queue_priority: int | None,
    workers_per_node: int | None,
    job_class: str | None,
) -> list[str]:
    assert generator_script.is_file(), generator_script
    assert heredoc_file.is_file(), heredoc_file
    assert launch_flag in {"--launch", "--launch-affinity"}, launch_flag
    assert launch_targets and result_root
    command = [
        "uv",
        "run",
        "python",
        str(generator_script),
        launch_flag,
        *launch_targets,
        "--heredoc-file",
        str(heredoc_file),
        "--heredoc-result-root",
        result_root,
    ]
    if queue_priority is not None:
        command.extend(["--queue-priority", str(queue_priority)])
    if workers_per_node is not None:
        command.extend(["--workers-per-node", str(workers_per_node)])
    if job_class is not None:
        command.extend(["--job-class", job_class])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a repo-local bash payload through a generator-shaped queue script "
            "and collect the worker logs."
        )
    )
    parser.add_argument(
        "--generator-script",
        type=Path,
        default=DEFAULT_GENERATOR_SCRIPT,
        help=(
            "Repo-local script that delegates to polyrepo_launch.executor.run(...)."
        ),
    )
    launch_target = parser.add_mutually_exclusive_group(required=True)
    launch_target.add_argument("--launch", nargs="+")
    launch_target.add_argument("--launch-affinity", nargs="+")
    parser.add_argument(
        "--heredoc-file",
        "--heredoc_file",
        type=Path,
        help="Repo-local payload script. Omit to read the payload from stdin.",
    )
    parser.add_argument(
        "--lines",
        type=int,
        required=True,
        help="How many trailing log lines to print per worker result file.",
    )
    parser.add_argument(
        "--print-only-worker",
        type=int,
        help="Still wait for every worker, but print only this rank's log.",
    )
    parser.add_argument("--queue-priority", "--priority", type=int)
    parser.add_argument("--workers-per-node", "--n-workers", type=int)
    parser.add_argument("--job-class", choices=("regular", "idle"))
    args = parser.parse_args()

    root = workspace_root()
    run_id = new_artifact_id()
    generator_script = repo_local_file(args.generator_script, root)
    stdin_payload = args.heredoc_file is None
    if stdin_payload:
        assert not sys.stdin.isatty(), "no --heredoc-file and no heredoc on stdin"
        heredoc_file = root / f"heredoc_stdin_{run_id}.sh"
        heredoc_file.write_text(sys.stdin.read())
        print(f"=== Wrote stdin payload to {heredoc_file} ===")
        payload_stem = "stdin"
    else:
        heredoc_file = repo_local_file(args.heredoc_file, root)
        payload_stem = heredoc_file.stem

    launch_flag = "--launch" if args.launch is not None else "--launch-affinity"
    launch_targets = args.launch if args.launch is not None else args.launch_affinity
    launch_node_ids = _parse_launch_nodes(launch_targets)
    gcs_roots = resolve_gcs_roots_for_node_ids(launch_node_ids)
    result_name = f"{generator_script.stem}_{payload_stem}_{run_id}"
    result_root = f"{gcs_roots.write_root}/runs/heredoc_results/{result_name}"
    workers_per_node = (
        args.workers_per_node
        if args.workers_per_node is not None
        else infer_workers_per_node(launch_targets)
    )
    assert workers_per_node is not None, launch_targets

    command = generator_command(
        generator_script,
        launch_flag,
        launch_targets,
        heredoc_file,
        result_root,
        args.queue_priority,
        args.workers_per_node,
        args.job_class,
    )
    print(" ".join(command))
    try:
        subprocess.run(command, check=True)
    finally:
        if stdin_payload:
            heredoc_file.unlink()

    result_paths = [
        CloudPath(result_root) / f"{rank}.txt"
        for rank in range(workers_per_node)
    ]
    print(f"=== Waiting for {len(result_paths)} worker result file(s) under {result_root} ===")
    while True:
        missing = [path for path in result_paths if not path.exists()]
        if not missing:
            break
        print(f"  {len(result_paths) - len(missing)}/{len(result_paths)} done; polling again in {POLL_SECONDS}s")
        time.sleep(POLL_SECONDS)

    worker_results = []
    for path in result_paths:
        lines = path.read_text().splitlines()
        exit_line = next(
            line for line in reversed(lines)
            if line.startswith("HEREDOC_EXIT_CODE=")
        )
        exit_code = int(exit_line.partition("=")[2])
        worker_results.append((path, lines, exit_code))

    displayed_results = worker_results
    if args.print_only_worker is not None:
        displayed_results = [worker_results[args.print_only_worker]]
    for path, lines, _ in displayed_results:
        print(f"\n=== {path} (last {min(args.lines, len(lines))} of {len(lines)} lines) ===")
        print("\n".join(lines[-args.lines:]))

    failed_workers = [
        (path.name, exit_code)
        for path, _, exit_code in worker_results
        if exit_code != 0
    ]
    if failed_workers:
        raise RuntimeError(f"heredoc payload failures: {failed_workers}")


if __name__ == "__main__":
    main()

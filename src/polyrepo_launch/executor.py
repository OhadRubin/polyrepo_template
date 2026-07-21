"""
- description:
    Parse generated-training execution modes, select tasks for the chosen mode,
    and hand the queue generator a workload renderer supplied by the caller.
- usage:
    # Used by a project-local generator script; render one selected task locally.
    uv run python scripts/make_training_scripts.py --dry-run --launch v6e-16-node-1 --only-run '<task-name>'

    # Render a repo-local payload wrapper.
    uv run python scripts/make_training_scripts.py --dry-run --launch v6e-16-node-1 --heredoc-file setup.sh --heredoc-result-root gs://example-bucket/runs/heredoc_results/check
- user_story:
    content:
        Ohad wants command-line execution modes to stay independent of the
        experiment-specific shell template, so the executor turns the selected
        mode into a workload request and the script renders that request into
        its shared shell template.
    was_generated_via_skill: false
"""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from . import dag
from .gcs_config import GcsRoots, workspace_root
from .generator import (
    RenderSpec,
    RenderedScript,
    main as generator_main,
    render_script,
)

Task = tuple[str, str]


HEREDOC_PREAMBLE = r'''
HEREDOC_LOG="$(mktemp)"
exec > >(tee -a "$HEREDOC_LOG") 2>&1
WORKER_RANK=$(curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number")
export WORKER_RANK
export HEREDOC_RESULT_ROOT="__RESULT_ROOT__"
HEREDOC_RESULT_URI="__RESULT_ROOT__/${WORKER_RANK}.txt"
echo "[$(hostname) rank=$WORKER_RANK] heredoc harness starting"
upload_heredoc_failure() {
  code=$?
  sleep 2
  echo "HEREDOC_EXIT_CODE=$code" >> "$HEREDOC_LOG"
  gcloud storage cp "$HEREDOC_LOG" "$HEREDOC_RESULT_URI"
  exit "$code"
}
trap upload_heredoc_failure ERR
'''


HEREDOC_HARNESS = r'''
cd "$REPO_DIR"
set +e
bash __PAYLOAD_PATH__
HEREDOC_STATUS=$?
set -e
trap - ERR
echo "[rank=$WORKER_RANK] payload finished with exit code $HEREDOC_STATUS"
sleep 2
echo "HEREDOC_EXIT_CODE=$HEREDOC_STATUS" >> "$HEREDOC_LOG"
gcloud storage cp "$HEREDOC_LOG" "$HEREDOC_RESULT_URI"
exit "$HEREDOC_STATUS"
'''


@dataclass(frozen=True)
class WorkloadRenderRequest:
    task: Task
    filename: str
    preamble: str
    workload_command: str


class ExecutionMode(Protocol):
    def select_plan(self, plan: tuple[Task, ...]) -> tuple[Task, ...]:
        raise NotImplementedError

    def render_request(
        self,
        task: Task,
        normal_workload_command: str,
    ) -> WorkloadRenderRequest:
        raise NotImplementedError


@dataclass(frozen=True)
class ExecutionArgs:
    heredoc_file: Path | None
    heredoc_result_root: str | None
    only_runs: tuple[str, ...]


class ExecutionModeFactory(Protocol):
    def __call__(self, args: ExecutionArgs) -> ExecutionMode:
        raise NotImplementedError


class ExperimentBuilder(Protocol):
    def __call__(self) -> dag.DAG:
        raise NotImplementedError


class RuntimeArgsRenderer(Protocol):
    def __call__(self, task: dict[str, object]) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class ExperimentRegistry:
    builders: dict[int, ExperimentBuilder]

    def register(
        self,
        *exp_counts: int,
    ) -> Callable[[ExperimentBuilder], ExperimentBuilder]:
        assert exp_counts

        def decorator(builder: ExperimentBuilder) -> ExperimentBuilder:
            overlap = [
                exp_count
                for exp_count in exp_counts
                if exp_count in self.builders
            ]
            assert not overlap, overlap
            for exp_count in exp_counts:
                self.builders[exp_count] = builder
            return builder

        return decorator

    def build(self, exp_count: int) -> dag.DAG:
        assert exp_count in self.builders, exp_count
        return self.builders[exp_count]()


@dataclass(frozen=True)
class NormalExecutionMode:
    only_runs: tuple[str, ...]

    def select_plan(self, plan: tuple[Task, ...]) -> tuple[Task, ...]:
        assert plan
        return select_plan(plan, self.only_runs)

    def render_request(
        self,
        task: Task,
        normal_workload_command: str,
    ) -> WorkloadRenderRequest:
        name, exports = task
        assert name and exports and normal_workload_command
        return WorkloadRenderRequest(
            task=task,
            filename=f"run_{name}.sh",
            preamble="",
            workload_command=normal_workload_command,
        )


@dataclass(frozen=True)
class HeredocExecutionMode:
    filename: str
    preamble: str
    workload_command: str

    def select_plan(self, plan: tuple[Task, ...]) -> tuple[Task, ...]:
        assert plan
        return plan[:1]

    def render_request(
        self,
        task: Task,
        normal_workload_command: str,
    ) -> WorkloadRenderRequest:
        name, exports = task
        assert name and exports and normal_workload_command
        return WorkloadRenderRequest(
            task=task,
            filename=self.filename,
            preamble=self.preamble,
            workload_command=self.workload_command,
        )


def normal_execution_mode(args: ExecutionArgs) -> ExecutionMode:
    assert args.heredoc_file is None
    return NormalExecutionMode(only_runs=args.only_runs)


def heredoc_execution_mode(args: ExecutionArgs) -> ExecutionMode:
    assert args.heredoc_file is not None
    assert args.heredoc_result_root
    assert not args.only_runs, "heredoc payloads use the default task"
    payload_rel = args.heredoc_file.relative_to(workspace_root())
    preamble = HEREDOC_PREAMBLE.replace("__RESULT_ROOT__", args.heredoc_result_root)
    workload_command = HEREDOC_HARNESS.replace(
        "__PAYLOAD_PATH__", shlex.quote(str(payload_rel))
    )
    return HeredocExecutionMode(
        filename=f"heredoc_{args.heredoc_file.stem}.sh",
        preamble=preamble,
        workload_command=workload_command,
    )


EXECUTION_MODE_FACTORIES: dict[str, ExecutionModeFactory] = {
    "normal": normal_execution_mode,
    "heredoc": heredoc_execution_mode,
}


def _pop_execution_args() -> ExecutionArgs:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--heredoc-file", "--heredoc_file", type=Path)
    parser.add_argument("--heredoc-result-root")
    parser.add_argument("--only-run", action="append", nargs="+")
    known, rest = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *rest]
    assert known.heredoc_file is not None or known.heredoc_result_root is None, (
        "heredoc result roots belong to heredoc file mode"
    )
    only_runs = (
        ()
        if known.only_run is None
        else tuple(
            name.strip()
            for group in known.only_run
            for name in group
            if name.strip()
        )
    )
    assert known.only_run is None or only_runs, known.only_run
    heredoc_file = (
        None
        if known.heredoc_file is None
        else known.heredoc_file.resolve()
    )
    return ExecutionArgs(
        heredoc_file=heredoc_file,
        heredoc_result_root=known.heredoc_result_root,
        only_runs=only_runs,
    )


def select_plan(plan: tuple[Task, ...], only_runs: tuple[str, ...]) -> tuple[Task, ...]:
    if not only_runs:
        return plan
    tasks_by_name = {task[0]: task for task in plan}
    missing = [name for name in only_runs if name not in tasks_by_name]
    assert not missing, missing
    return tuple(tasks_by_name[name] for name in only_runs)


def _pop_execution_mode() -> ExecutionMode:
    args = _pop_execution_args()
    mode_name = "heredoc" if args.heredoc_file is not None else "normal"
    return EXECUTION_MODE_FACTORIES[mode_name](args)


def render_workload(
    request: WorkloadRenderRequest,
    setup: str,
    gcs_roots: GcsRoots,
    exp_count: int,
    train_template: str,
) -> RenderedScript:
    name, exports = request.task
    assert name and exports and exp_count and train_template
    return render_script(
        RenderSpec(
            filename=request.filename,
            fragments=(request.preamble, setup, exports, train_template),
            variables={
                "EXP": exp_count,
                "READ_GCS_ROOT": gcs_roots.read_root,
                "WRITE_GCS_ROOT": gcs_roots.write_root,
                "WORKLOAD_COMMAND": request.workload_command,
            },
        )
    )


def stop_at_task(task: object) -> bool:
    assert task is not None
    return False


def make_plan(
    registry: ExperimentRegistry,
    exp_count: int,
    config: object,
    runtime_args: RuntimeArgsRenderer,
) -> tuple[Task, ...]:
    experiment = registry.build(exp_count)
    task_dict, odict = dag.get_all_experiments(
        experiment,
        config,
        exp_count,
        stop_at_task,
    )
    plan = tuple(
        (name, "\n".join((task_dict[name], runtime_args(odict[name]))))
        for name in task_dict
    )
    assert plan, exp_count
    return plan


def run(
    registry: ExperimentRegistry,
    exp_count: int,
    config: object,
    runtime_args: RuntimeArgsRenderer,
    train_template: str,
    normal_workload_command: str,
) -> None:
    plan = make_plan(registry, exp_count, config, runtime_args)
    assert plan and exp_count and train_template and normal_workload_command
    mode = _pop_execution_mode()
    selected_plan = mode.select_plan(plan)
    assert selected_plan

    def task_renderer(task: Task, setup: str, gcs_roots: GcsRoots) -> RenderedScript:
        return render_workload(
            mode.render_request(task, normal_workload_command),
            setup,
            gcs_roots,
            exp_count,
            train_template,
        )

    generator_main(selected_plan, task_renderer)

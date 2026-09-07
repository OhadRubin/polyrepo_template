"""
- description:
    Parse generated-training execution modes, select tasks for the chosen mode,
    and hand the queue generator a workload renderer supplied by the caller.
- usage:
    # Used by a project-local generator script; render every minor of plan 113, patch 0.
    uv run python scripts/make_training_scripts.py --dry-run --launch v6e-16-node-1 --exp 113

    # Minors 1 and 2 only (shell brace expansion), relaunched under a fresh identity (patch 2).
    uv run python scripts/make_training_scripts.py --dry-run --launch v6e-16-node-1 --exp 113 --minor {1..2} --patch 2

    # One cell, by its parameter-derived name, across the selected minors.
    uv run python scripts/make_training_scripts.py --dry-run --launch v6e-16-node-1 --exp 113 --only-run '<cell-name>'

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
import functools
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

@dataclass(frozen=True)
class Task:
    """One launchable cell under its full identity v{exp}.{minor}.{patch}_{cell}."""
    exp: int
    minor: int
    patch: int
    cell: str
    exports: str

    @property
    def run_id(self) -> str:
        return f"v{self.exp}.{self.minor}.{self.patch}_{self.cell}"


@dataclass(frozen=True)
class LaunchSelection:
    """`--exp`, `--minor` (empty selects every emitted minor) and `--patch`."""
    exps: tuple[int, ...]
    minors: tuple[int, ...]
    patch: int


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


def shell_array(name: str, args: list[str]) -> str:
    """Renders a bash array assignment for a runtime_args fragment."""
    if not args:
        return f"{name}=()"
    return "\n".join((
        f"{name}=(",
        *(f'  "{arg}"' for arg in args),
        ")",
    ))


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

    def plan(
        self,
        *exp_counts: int,
    ) -> Callable[[Callable[[], None]], Callable[[], None]]:
        """Registers an `add_*` sweep function as the numbered plan(s),
        absorbing the `with dag.DAG(): ...` builder boilerplate. Returns
        the add function unchanged so plans can still compose each other."""

        def wrap(add_fn: Callable[[], None]) -> Callable[[], None]:
            @self.register(*exp_counts)
            @functools.wraps(add_fn)
            def build() -> dag.DAG:
                with dag.DAG() as experiment:
                    add_fn()
                return experiment

            return add_fn

        return wrap

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
        assert task.exports and normal_workload_command
        return WorkloadRenderRequest(
            task=task,
            filename=f"run_{task.run_id}.sh",
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
        assert task.exports and normal_workload_command
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
    """Keeps the named cells in every selected minor."""
    if not only_runs:
        return plan
    selected = tuple(task for task in plan if task.cell in only_runs)
    missing = set(only_runs) - {task.cell for task in selected}
    assert not missing, missing
    return selected


def _pop_execution_mode() -> ExecutionMode:
    args = _pop_execution_args()
    mode_name = "heredoc" if args.heredoc_file is not None else "normal"
    return EXECUTION_MODE_FACTORIES[mode_name](args)


def _pop_launch_selection(registry: ExperimentRegistry, heredoc: bool) -> LaunchSelection:
    parser = argparse.ArgumentParser(add_help=False)
    # `nargs="+"` lets the shell's `{A..B}` expansion pass several values.
    parser.add_argument("--exp", type=int, nargs="+")
    parser.add_argument("--minor", type=int, nargs="+")
    parser.add_argument("--patch", type=int, default=0)
    known, rest = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *rest]
    if known.exp is None:
        # a heredoc payload replaces the plan, so which plan carries it is
        # immaterial; normal launches must say which plan they run
        assert heredoc, "--exp <N> is required (which registered plan to run)"
        exps = (min(registry.builders),)
    else:
        exps = tuple(known.exp)
    return LaunchSelection(
        exps=exps,
        minors=() if known.minor is None else tuple(known.minor),
        patch=known.patch,
    )


def render_workload(
    request: WorkloadRenderRequest,
    setup: str,
    gcs_roots: GcsRoots,
    train_template: str,
) -> RenderedScript:
    task = request.task
    assert task.exports and train_template
    return render_script(
        RenderSpec(
            filename=request.filename,
            fragments=(request.preamble, setup, task.exports, train_template),
            variables={
                "EXP": task.exp,
                "MINOR": task.minor,
                "PATCH": task.patch,
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
    selection: LaunchSelection,
    config: object,
    runtime_args: RuntimeArgsRenderer,
) -> tuple[Task, ...]:
    plan = tuple(
        Task(
            exp=exp,
            minor=cell.minor,
            patch=selection.patch,
            cell=cell.name,
            exports="\n".join((
                f"export WANDB_NAME=v{exp}.{cell.minor}.{selection.patch}_{cell.name}",
                f"export POLYREPO_EXP={exp}",
                f"export POLYREPO_MINOR={cell.minor}",
                f"export POLYREPO_PATCH={selection.patch}",
                f'export WANDB_TAGS="v{exp},v{exp}.{cell.minor},'
                f'v{exp}.{cell.minor}.{selection.patch},'
                f'v{exp}.X.{selection.patch}"',
                cell.exports,
                runtime_args(cell.arg_vars),
            )),
        )
        for exp in selection.exps
        for cell in dag.get_all_experiments(registry.build(exp), config, stop_at_task)
        if not selection.minors or cell.minor in selection.minors
    )
    missing_minors = set(selection.minors) - {task.minor for task in plan}
    assert plan and not missing_minors, (selection, missing_minors)
    return plan


def run(
    registry: ExperimentRegistry,
    config: object,
    runtime_args: RuntimeArgsRenderer,
    train_template: str,
    normal_workload_command: str,
) -> None:
    mode = _pop_execution_mode()
    selection = _pop_launch_selection(
        registry, isinstance(mode, HeredocExecutionMode)
    )
    plan = make_plan(registry, selection, config, runtime_args)
    assert train_template and normal_workload_command
    selected_plan = mode.select_plan(plan)
    assert selected_plan

    def task_renderer(task: Task, setup: str, gcs_roots: GcsRoots) -> RenderedScript:
        return render_workload(
            mode.render_request(task, normal_workload_command),
            setup,
            gcs_roots,
            train_template,
        )

    generator_main(selected_plan, task_renderer)

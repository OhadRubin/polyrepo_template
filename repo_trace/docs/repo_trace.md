# Trace an existing command

`repo-trace` records Python definitions reached by a command inside the current
Git repository. It requires Python 3.12 or newer and `uv`. Run it from anywhere
inside the target repository:

```bash
uv run /path/to/repo_trace/repo-trace -- uv run python your_program.py arg1 arg2
```

Replace `/path/to/repo_trace` with the directory containing this tool.
The tool directory and the target repository can be in separate locations.
The command keeps its working directory, arguments, environment, and terminal
output. `repo-trace` detects the Git root and saves evidence under a unique
`results/repo-trace/` directory. Use `--output DIRECTORY` before `--` to select
an unused output directory. It writes a report after the command exits,
including when the command returns an error, and returns the command's status.

## What to read

| File under the output directory | Contents |
| --- | --- |
| `report/functions.tsv` | Named functions observed during workload execution |
| `report/imports.tsv` | Definitions observed while importing modules |
| `report/definitions.tsv` | Module bodies, class bodies, and expressions |
| `report/manifest.json` | Observed stacks, source locations, dependencies, process membership, and command results |
| `report/report.md` | Counts by source file and limits of the evidence |
| `calls/` | Raw observations, persisted as they happen |
| `sources/` | Copies of the observed source files |
| `run.json` | Command, working directory, repository root, and exit code |

A function called during imports remains observable until it is also called
outside imports. It can appear in both tables. Class definitions encountered
during registration are available in the supporting views. This keeps the
main function list focused on workload execution.

The hook uses Python's `sitecustomize` startup convention and preserves an
existing `sitecustomize` from the target environment. It records the first
import and first workload context for each code object in each process,
including the surrounding Python stack through external libraries. After
workload observation, the hook disables further events for that location to
keep repeated computation inexpensive. Generator resumptions are observed too.

Import attribution detects importlib loading and the main-module initialization
performed by multiprocessing spawn. A lazily imported module is still import
activity, even when imported halfway through execution. Its side effects can
be required dependencies and need inspection before extraction.

## Threads, workers, and JAX

Threads share the interpreter hook. Python subprocesses inherit it through
`PYTHONPATH`; a fork resets the child's inventory. Each record includes its
process and parent process IDs. Stack evidence belongs to the executing thread
or worker; it does not reconstruct the task submission stack in another thread
or process.

JAX tracing exposes the Python used to build compiled computations. Start the
wrapper before compilation. Staging may visit multiple compiled branches, and
Python observations do not identify which device branch subsequently executes.

Child commands that clear the environment or disable Python startup hooks
(`-I`, `-E`, or `-S`) need integration at their launch boundary. Native function
bodies and generated code with synthetic filenames are outside the inventory.

## Queued probes

The wrapper recognizes `polyrepo-heredoc` and stages the hook with its payload.
Use the ordinary launcher arguments; both `--heredoc-file` and a payload on
standard input are supported:

```bash
uv run /path/to/repo_trace/repo-trace -- uv run polyrepo-heredoc \
    --generator-script launcher.py --launch-affinity "$WORKER_NAME" \
    --heredoc-file probe.sh --lines 100
```

Use the target project's launcher, payload script, and worker name.
For long probes, tmux and `tee` retain the session and its output.
The wrapper submits once through the existing launcher. Each worker
runs the supplied payload under the generic hook, preserving its shell traps.
It saves traces and source copies in `/dev/shm` and uploads them beneath the
heredoc result directory's `repo-trace/rank-N/`. The adapter collects these under
`workers/` and builds a combined local report. `launcher.log` retains queue
output and `run.json` records the remote location. Uploads include evidence from
ordinary command failures. Abrupt worker termination can prevent an upload.

The adapter uses the result location printed by the installed
`polyrepo-heredoc` client. Other remote launchers need their own transport
integration; ordinary wrapping observes the local processes they start.

## Using the evidence for extraction

The manifest retains referenced globals, decorators, module bindings, source
imports, and enclosing classes/functions for workload definitions. These are
dependency evidence for assembling an extracted program. Dynamic registrations,
reflective lookups, and import side effects still need inspection.

Logging, checkpointing, and reporting functions called by the workload appear
in the workload inventory. Their inclusion in the extracted program depends
on its required behavior. The tool does not infer that intent from execution.
Verify an extraction by running controlled examples with the original
repository unavailable and comparing the results.

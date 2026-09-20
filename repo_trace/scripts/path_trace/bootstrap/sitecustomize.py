"""
- description:
    Install tracing in Python interpreters started by the wrapper.
- usage:
    uv run ./repo-trace -- uv run python your_program.py
    # The wrapper adds this directory to PYTHONPATH automatically.
- user_story:
    content:
        Ohad can trace subprocesses using Python's startup convention while
        preserving the target environment's existing startup customization.
    was_generated_via_skill: false
"""

# Python imports this module before the workload and in spawned workers.
import os

if 'REPO_TRACE_ROOT' in os.environ:
  import importlib.machinery
  import importlib.util
  from pathlib import Path
  import sys
  import traceback

  try:
    hook_dir = Path(__file__).resolve().parent
    recorder_spec = importlib.util.spec_from_file_location(
        '_repo_trace_recorder', hook_dir.parent / 'recorder.py',
    )
    recorder = importlib.util.module_from_spec(recorder_spec)
    sys.modules[recorder_spec.name] = recorder
    recorder_spec.loader.exec_module(recorder)

    # Preserve sitecustomize supplied by the target environment.
    search = [p for p in sys.path if Path(p).resolve() != hook_dir]
    spec = importlib.machinery.PathFinder.find_spec('sitecustomize', search)
    if spec is not None:
      spec.loader.exec_module(importlib.util.module_from_spec(spec))
  except Exception:
    # Python normally prints startup errors and continues without the hook.
    # An incomplete trace must fail visibly instead of looking successful.
    traceback.print_exc()
    os._exit(1)

"""
- description:
    Record repository calls with import context and surrounding stacks.
- usage:
    uv run ./repo-trace -- uv run python your_program.py
    # The wrapper loads this recorder through Python's startup hook.
- user_story:
    content:
        Ohad can distinguish import initialization from workload execution
        while preserving callbacks, threads, and child process evidence.
    was_generated_via_skill: false
"""

import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import threading


ROOT = Path(os.environ['REPO_TRACE_ROOT']).resolve()
OUTPUT = Path(os.environ['REPO_TRACE_OUTPUT'])
TOOL_DIR = Path(__file__).resolve().parent
(OUTPUT / 'calls').mkdir(parents=True, exist_ok=True)
LOCK = threading.RLock()
FILES = {}
SEEN = set()
SOURCES = set()
TOOL_ID = 3


@functools.cache
def source_location(code):
  """Identify code by its source, preserving generated-code filenames."""
  filename = code.co_filename
  if not filename.startswith('<'):
    path = Path(filename).resolve()
    filename = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) \
        else str(path)
  return filename, code.co_firstlineno, code.co_qualname


@functools.cache
def repository_code(code):
  filename = code.co_filename
  if filename.startswith('<') or not filename.endswith('.py'):
    return False
  path = Path(filename).resolve()
  return (path.is_relative_to(ROOT) and not path.is_relative_to(TOOL_DIR)
          and not {'.venv', 'site-packages'}.intersection(path.parts))


def record_entry(code, offset):
  """Retain import evidence until a function also enters the workload."""
  if not repository_code(code):
    # Disabling library entry events removes their recurring tracing cost.
    return sys.monitoring.DISABLE
  callee = source_location(code)
  frame = sys._getframe(1)
  parent = frame.f_back
  caller = source_location(parent.f_code) if parent is not None else None
  call_line = parent.f_lineno if parent is not None else None
  stack = []
  origin = 'workload'
  cursor = frame
  while cursor is not None:
    module = cursor.f_globals.get('__name__', '')
    name = cursor.f_code.co_name
    # Spawn re-executes __main__ through runpy rather than importlib.
    if (module in ('importlib._bootstrap', 'importlib._bootstrap_external')
        or (module == 'multiprocessing.spawn'
            and name in ('_fixup_main_from_path', '_fixup_main_from_name'))):
      origin = 'import'
    stack.append({
        'function': source_location(cursor.f_code),
        'line': cursor.f_lineno, 'module': module,
        'repository': repository_code(cursor.f_code),
    })
    cursor = cursor.f_back
  pid = os.getpid()
  observation = (pid, callee, origin)
  with LOCK:
    if observation not in SEEN:
      SEEN.add(observation)
      if pid not in FILES:
        path = OUTPUT / 'calls' / f'{os.uname().nodename}-{pid}.jsonl'
        FILES[pid] = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644,
        )
      records = []
      filename = callee[0]
      if (pid, filename) not in SOURCES:
        SOURCES.add((pid, filename))
        source = (ROOT / filename).read_bytes()
        destination = OUTPUT / 'sources' / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f'{destination.name}.{pid}.tmp')
        temporary.write_bytes(source)
        temporary.replace(destination)
        records.append({
            'kind': 'source', 'path': filename,
            'sha256': hashlib.sha256(source).hexdigest(),
        })
      records.append({
          'kind': 'call', 'callee': callee, 'caller': caller,
          'call_line': call_line, 'thread': threading.current_thread().name,
          'module': frame.f_globals['__name__'], 'origin': origin,
          'stack': stack, 'pid': pid, 'parent_pid': os.getppid(),
      })
      # Persist new evidence immediately: process pools can use os._exit.
      payload = ''.join(json.dumps(row) + '\n' for row in records).encode()
      os.write(FILES[pid], payload)
  # Inventory membership is known after one workload entry. Let frequently
  # called helpers run at normal speed for the remaining execution.
  if origin == 'workload':
    return sys.monitoring.DISABLE


def after_fork():
  """A fork starts a new inventory even for code seen in its parent."""
  global LOCK
  LOCK = threading.RLock()
  for descriptor in FILES.values():
    os.close(descriptor)
  FILES.clear()
  SEEN.clear()
  SOURCES.clear()
  sys.monitoring.restart_events()


sys.monitoring.use_tool_id(TOOL_ID, 'repository-call-inventory')
for event in (sys.monitoring.events.PY_START, sys.monitoring.events.PY_RESUME):
  sys.monitoring.register_callback(TOOL_ID, event, record_entry)
if hasattr(os, 'register_at_fork'):
  os.register_at_fork(after_in_child=after_fork)
sys.monitoring.set_events(
    TOOL_ID, sys.monitoring.events.PY_START | sys.monitoring.events.PY_RESUME,
)

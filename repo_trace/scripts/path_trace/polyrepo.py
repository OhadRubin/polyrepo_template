"""
- description:
    Ship the tracer with a polyrepo heredoc and collect worker evidence.
- usage:
    uv run ./repo-trace -- uv run polyrepo-heredoc [launcher arguments]
    # Normal polyrepo-heredoc arguments select the payload and worker pool.
- user_story:
    content:
        Ohad can wrap a queued probe with the same executable used locally
        and receive an inventory of its workers' actual execution.
    was_generated_via_skill: false
"""

import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile


def trace_heredoc(command, root, output, metadata):
  tool_dir = Path(__file__).resolve().parent
  # Polyrepo syncs this small bundle through its ordinary repository upload.
  # The payload runs in a child shell, preserving its own traps and exit code.
  with tempfile.TemporaryDirectory(prefix='repo_trace_', dir=root) as staging:
    stage = Path(staging)
    for name in ('cli.py', 'recorder.py', 'report.py'):
      shutil.copy2(tool_dir / name, stage / name)
    shutil.copytree(
        tool_dir / 'bootstrap', stage / 'bootstrap',
        ignore=shutil.ignore_patterns('__pycache__'),
    )
    launch = list(command)
    payload = None
    for index, argument in enumerate(launch):
      key, separator, value = argument.partition('=')
      if key in ('--heredoc-file', '--heredoc_file'):
        if separator:
          payload = Path(value).read_text()
          launch[index] = f'{key}={stage / "launch.sh"}'
        else:
          payload = Path(launch[index + 1]).read_text()
          launch[index + 1] = str(stage / 'launch.sh')
        break
    if payload is None:
      payload = sys.stdin.read()
      launch.extend(['--heredoc-file', str(stage / 'launch.sh')])
    (stage / 'payload.sh').write_text(payload)
    relative = shlex.quote(str(stage.relative_to(root)))
    (stage / 'launch.sh').write_text(f'''#!/usr/bin/env bash
set -uo pipefail
cd "$REPO_DIR"
trace_dir="/dev/shm/repo-trace/${{HEREDOC_RESULT_ROOT##*/}}/rank-$WORKER_RANK"
uv run --active python {relative}/cli.py --output "$trace_dir" -- \\
    bash {relative}/payload.sh
trace_status=$?
gcloud storage cp --recursive "$trace_dir" "$HEREDOC_RESULT_ROOT/repo-trace/"
upload_status=$?
if (( upload_status != 0 )); then exit "$upload_status"; fi
exit "$trace_status"
''')
    remote_root = None
    with (output / 'launcher.log').open('w') as log:
      with subprocess.Popen(
          launch, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
          text=True, bufsize=1,
      ) as process:
        for line in process.stdout:
          print(line, end='', flush=True)
          log.write(line)
          log.flush()
          match = re.search(
              r'Waiting for \d+ worker result file\(s\) under (gs://\S+)',
              line,
          )
          if match:
            remote_root = match.group(1)
        exit_code = process.wait()
  metadata.update(exit_code=exit_code, remote_root=remote_root)
  (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
  if remote_root is not None:
    subprocess.run([
        'gcloud', 'storage', 'rsync', '--recursive',
        f'{remote_root}/repo-trace', str(output / 'workers'),
    ], check=True)
    for sources in sorted((output / 'workers').glob('rank-*/sources')):
      shutil.copytree(sources, output / 'sources', dirs_exist_ok=True)
    subprocess.run([
        sys.executable, str(tool_dir / 'report.py'),
        '--traces', str(output / 'workers'),
        '--repo', str(output / 'sources'), '--output', str(output / 'report'),
    ], check=True)
    print(f'Report: {output / "report" / "report.md"}', flush=True)
  return exit_code if exit_code >= 0 else 128 - exit_code

"""
- description:
    Run an existing command with repository tracing and build its report.
- usage:
    uv run ./repo-trace -- uv run python your_program.py
    # Detect the Git root and save evidence under results/repo-trace/.
- user_story:
    content:
        Ohad can inspect a program's dependencies by wrapping its usual
        command, including Python workers, without declaring entry points.
    was_generated_via_skill: false
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def main():
  parser = argparse.ArgumentParser(
      prog='repo-trace',
      description='Trace a command and report repository dependencies.',
  )
  parser.add_argument('--output', type=Path)
  parser.add_argument('command', nargs=argparse.REMAINDER)
  args = parser.parse_args()
  command = args.command
  if command[:1] == ['--']:
    command = command[1:]
  root = Path(subprocess.check_output(
      ['git', 'rev-parse', '--show-toplevel'], text=True,
  ).strip())
  stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
  output = (args.output or root / 'results' / 'repo-trace'
            / f'{stamp}-{uuid.uuid4().hex[:8]}').resolve()
  output.mkdir(parents=True, exist_ok=True)
  # Exclusive creation prevents two commands from mixing their evidence.
  metadata = {'command': command, 'cwd': str(Path.cwd()), 'repo': str(root)}
  with (output / 'run.json').open('x') as stream:
    json.dump(metadata, stream, indent=2)
  print(f'Repository trace: {output}', flush=True)
  tool_dir = Path(__file__).resolve().parent
  executable = Path(command[0]).name
  is_heredoc = executable == 'polyrepo-heredoc' or (
      executable == 'uv' and command[1:2] == ['run']
      and any(Path(arg).name == 'polyrepo-heredoc' for arg in command[2:])
  )
  if is_heredoc:
    from polyrepo import trace_heredoc

    return trace_heredoc(command, root, output, metadata)

  python_path = [str(tool_dir / 'bootstrap')]
  if 'PYTHONPATH' in os.environ:
    python_path.append(os.environ['PYTHONPATH'])
  env = os.environ | {
      'REPO_TRACE_ROOT': str(root),
      'REPO_TRACE_OUTPUT': str(output),
      'PYTHONPATH': os.pathsep.join(python_path),
  }
  result = subprocess.run(command, env=env)
  metadata['exit_code'] = result.returncode
  (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
  # Reporting runs outside the traced environment, even for failed commands.
  subprocess.run([
      sys.executable, str(tool_dir / 'report.py'),
      '--traces', str(output), '--repo', str(output / 'sources'),
      '--output', str(output / 'report'),
  ], check=True)
  print(f'Report: {output / "report" / "report.md"}', flush=True)
  code = result.returncode
  return code if code >= 0 else 128 - code


if __name__ == '__main__':
  sys.exit(main())

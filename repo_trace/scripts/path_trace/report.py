"""
- description:
    Merge process traces into a source inventory and extraction manifest.
- usage:
    uv run python scripts/path_trace/report.py --help
    # Read the collected traces and resolve their definitions in this repo.
- user_story:
    content:
        Ohad can inspect which definitions executed, their callers, and
        the source files that need attention when extracting a workload.
    was_generated_via_skill: false
"""

# Runtime evidence and source dependencies stay separately inspectable.
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import symtable


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--traces', type=Path, required=True)
  parser.add_argument('--repo', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  functions = collections.defaultdict(set)
  origins = collections.defaultdict(lambda: collections.defaultdict(set))
  hashes = collections.defaultdict(set)
  edges = set()
  calls = []
  processes = []
  runs = {
      str(path.relative_to(args.traces)): json.loads(path.read_text())
      for path in sorted(args.traces.rglob('run.json'))
  }
  for path in sorted(args.traces.rglob('calls/*.jsonl')):
    process = str(path.relative_to(args.traces))
    processes.append(process)
    for line in path.read_text().splitlines():
      record = json.loads(line)
      if record['kind'] == 'source':
        hashes[record['path']].add(record['sha256'])
      else:
        callee = tuple(record['callee'])
        caller = tuple(record['caller']) if record['caller'] else None
        functions[callee].add(process)
        origins[callee][record['origin']].add(process)
        edges.add((callee, caller, record['call_line'], record['origin']))
        calls.append(record | {'process': process})

  sources = {}
  units = {}
  imports = []
  for filename, recorded_hashes in sorted(hashes.items()):
    path = args.repo / filename
    source = path.read_bytes()
    current_hash = hashlib.sha256(source).hexdigest()
    tree = ast.parse(source, filename=filename)
    units[filename] = [
        (min([node.lineno] + [d.lineno for d in node.decorator_list]),
         node.end_lineno, node.name)
        for node in tree.body
        if isinstance(node, (
            ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        ))
    ]
    tables = [symtable.symtable(source, filename, 'exec')]
    symbols = {}
    while tables:
      table = tables.pop()
      symbols[table.get_lineno(), table.get_name()] = sorted(
          symbol.get_name() for symbol in table.get_symbols()
          if symbol.is_referenced() and symbol.is_global()
      )
      tables.extend(table.get_children())
    bindings = {}
    for node in tree.body:
      if isinstance(node, (
          ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
      )):
        bound_names = [node.name]
      elif isinstance(node, (ast.Import, ast.ImportFrom)):
        bound_names = [
            alias.asname or alias.name.split('.')[0] for alias in node.names
        ]
      elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        bound_names = [
            part.id for target in targets for part in ast.walk(target)
            if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)
        ]
      else:
        bound_names = []
      for name in bound_names:
        bindings[name] = {'line': node.lineno, 'end_line': node.end_lineno}
    definitions = {}
    for node in ast.walk(tree):
      if isinstance(node, (
          ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
      )):
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        definitions[start] = {
            'kind': 'class_body' if isinstance(node, ast.ClassDef)
            else 'function',
            'end_line': node.end_lineno,
            'referenced_globals': symbols[node.lineno, node.name],
            'decorators': [ast.unparse(d) for d in node.decorator_list],
        }
      elif isinstance(node, (ast.Import, ast.ImportFrom)):
        imports.append({
            'file': filename, 'line': node.lineno,
            'statement': ast.get_source_segment(source.decode(), node),
        })
    sources[filename] = {
        'recorded_sha256': sorted(recorded_hashes),
        'current_sha256': current_hash,
        'matches_current_source': recorded_hashes == {current_hash},
        'module_bindings': bindings,
        'has_user_story_docstring': 'user_story:' in (
            ast.get_docstring(tree) or ''
        ),
        'definitions': definitions,
    }

  entries = []
  source_units = collections.defaultdict(list)
  for (filename, start, name), visited_by in sorted(functions.items()):
    # Comprehensions and lambdas have code objects of their own; retain
    # them as evidence while identifying the named definitions separately.
    definition = sources[filename]['definitions'].get(start)
    kind = 'expression'
    end = None
    if name == '<module>':
      kind = 'module_body'
    elif definition is not None and not name.endswith('>'):
      kind, end = definition['kind'], definition['end_line']
    if kind == 'function' and 'workload' in origins[filename, start, name]:
      for unit_start, unit_end, unit_name in units[filename]:
        if unit_start <= start <= unit_end:
          source_units[filename, unit_start, unit_end, unit_name].append(name)
          break
    entries.append({
        'file': filename, 'line': start, 'name': name,
        'kind': kind, 'end_line': end, 'processes': sorted(visited_by),
        'origins': {
            origin: sorted(process_names) for origin, process_names
            in sorted(origins[filename, start, name].items())
        },
        'referenced_globals': (
            definition['referenced_globals'] if definition else []
        ),
        'decorators': definition['decorators'] if definition else [],
    })
  for source in sources.values():
    del source['definitions']
  manifest = {
      'processes': processes, 'sources': sources, 'entries': entries,
      'runs': runs,
      'calls': calls, 'source_imports': imports,
      'source_units': [
          {'file': key[0], 'line': key[1], 'end_line': key[2],
           'name': key[3], 'observed_functions': names}
          for key, names in sorted(source_units.items())
      ],
      'limits': [
          'Observed Python entries cover only the executions collected.',
          'JAX tracing exposes Python used to construct compiled programs.',
          'JAX staging can visit Python for multiple compiled branches.',
          'Generated filenames and native function bodies are excluded.',
          'Stacks retain first import and first workload context per code '
          'object and process; later callers are not enumerated.',
          'Import context includes importlib loading and multiprocessing '
          'spawn initialization; manually executed module code may differ.',
          'Threads and workers have their own stacks. Parent process IDs '
          'identify processes, not the operation that submitted a task.',
          'Runtime logging and checkpoint functions remain workload evidence; '
          'their necessity depends on the intended extracted behavior.',
          'Child interpreters must inherit PYTHONPATH and allow site startup. '
          'Python -I, -E, -S and replaced environments can bypass the hook.',
          'Imports are source references, including unexecuted branches.',
          'Function bodies may require globals, decorators, and registration.',
          'Global names use Python symbol tables; reflective lookups need '
          'separate inspection.',
          'Module bindings locate direct top-level declarations; conditional '
          'and dynamic bindings need separate inspection.',
          'Source units retain enclosing top-level classes and functions; '
          'module initialization still needs separate inspection.',
          'A source hash mismatch makes current AST ranges provisional.',
      ],
  }
  (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
  workload = [
      e for e in entries
      if e['kind'] == 'function' and 'workload' in e['origins']
  ]
  tables = {
      'functions': workload,
      'imports': [e for e in entries if 'import' in e['origins']],
      'definitions': [e for e in entries if e['kind'] != 'function'],
  }
  for label, selected in tables.items():
    rows = ['file\tline\tend_line\tname\tkind\torigins\tprocess_count']
    rows.extend(
        '\t'.join(map(str, (
            e['file'], e['line'], e['end_line'], e['name'], e['kind'],
            ','.join(e['origins']), len(e['processes']),
        ))) for e in selected
    )
    (args.output / f'{label}.tsv').write_text('\n'.join(rows) + '\n')
  counts = collections.Counter(
      e['file'] for e in workload
  )
  lines = [
      '# Observed repository execution', '',
      f'Processes with repository calls: {len(processes)}. '
      f'Workload functions: {sum(counts.values())}. '
      f'Unique observed caller edges: {len(edges)}.', '',
      f'Command results: {len(runs)}. '
      'Commands and exit codes are retained in `manifest.json`.', '',
      '`functions.tsv` lists named functions observed outside imports. '
      '`imports.tsv` retains import observations; `definitions.tsv` lists '
      'module/class bodies and expressions. These views can overlap.', '',
      '`manifest.json` contains source dependencies and the observed stacks, '
      'including external library frames. Source copies are saved by the '
      'recorder so the report can resolve the code that actually ran.', '',
      '| Source | Named functions | Source hash matches |',
      '| --- | ---: | --- |',
  ]
  for filename, count in counts.most_common():
    matches = sources[filename]['matches_current_source']
    lines.append(f'| {filename} | {count} | {matches} |')
  lines.extend(['', '## Scope of the evidence', ''])
  lines.extend(f'- {limit}' for limit in manifest['limits'])
  (args.output / 'report.md').write_text('\n'.join(lines) + '\n')
  print(json.dumps({
      'processes': len(processes), 'functions': sum(counts.values()),
      'command_results': len(runs),
      'files_with_functions': len(counts), 'edges': len(edges),
      'import_entries': len(tables['imports']),
      'source_mismatches': [
          name for name, value in sources.items()
          if not value['matches_current_source']
      ],
  }), flush=True)


if __name__ == '__main__':
  main()

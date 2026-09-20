# Repository tracing

A standalone executable for discovering which Python definitions an existing
command uses. Run it from the Git repository you want to inspect:

```bash
uv run /path/to/repo_trace/repo-trace -- uv run python your_program.py
```

Requires Python 3.12 or newer and `uv`. The tool includes import attribution,
thread and process tracing, source snapshots, report generation, and a
`polyrepo-heredoc` adapter for queued commands.

See the [usage guide](docs/repo_trace.md) for outputs and tracing limits.

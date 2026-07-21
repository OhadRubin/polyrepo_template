"""
- description:
    Package the command-line client for the local TPU dispatch HTTP API.
- usage:
    uv run queue-cli --help
    # Use queue-cli to enqueue, inspect, cancel, and reassign dispatch jobs.
- user_story:
    content:
        As the TPU queue operator, Ohad wants the dispatch CLI packaged as its
        own backend-facing tool so submission concerns stay separate from
        polyrepo workspace synchronization and launch script generation.
    was_generated_via_skill: false
"""

__all__ = []

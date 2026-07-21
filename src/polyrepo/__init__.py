"""
- description:
    Expose the manifest-driven polyrepo synchronization library used by launchers
    that publish several independent Git repositories as one remote workspace state.
- usage:
    uv run python -c "from polyrepo import sync_repo; print(sync_repo.publish)"
    # Import sync_repo from launcher code and publish a manifest-backed workspace.
- user_story:
    content:
        As a launcher author, I want one stable library namespace for polyrepo
        publication so that deployment code depends on a small shared contract.
    was_generated_via_skill: false
"""

from . import sync_repo
from .sync_repo import PublishedWorkspace

__all__ = ["PublishedWorkspace", "sync_repo"]

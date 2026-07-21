"""
- description:
    Create globally unique identifiers for immutable publication and launch
    artifacts shared by the sync and launcher packages.
- usage:
    uv run python -c "from polyrepo.artifact_id import new_artifact_id; print(new_artifact_id())"
    # Use the identifier as one component of a GCS object or run path.
- user_story:
    content:
        As a launcher author, I want every publication and launch to receive a
        unique artifact namespace so queued work always resolves the exact code
        and result objects created for that launch.
    was_generated_via_skill: false
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


def new_artifact_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex}"

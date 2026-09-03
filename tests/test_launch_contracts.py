"""
- description:
    Verify artifact identity, queue targeting, and manifest-backed queue upload
    behavior.
- usage:
    uv run python -m unittest discover -s tests -p 'test_launch_contracts.py'
    # Run the focused launch contract tests from the repository root.
- user_story:
    content:
        As a maintainer, I want focused tests around artifact identity, queue
        targeting, and one-off upload policy so launch isolation and configured
        storage remain stable as the public package changes.
    was_generated_via_skill: false
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from polyrepo.artifact_id import new_artifact_id
from polyrepo.sync_repo import PublicationLayout, publish
from polyrepo_launch import generator
from tpu_dispatch_cli.queue_cli import (
    QueueCliError,
    affinity_query,
    build_parser,
    run_script,
)


class LaunchContractTests(unittest.TestCase):
    def test_artifact_ids_are_unique(self) -> None:
        first = new_artifact_id()
        second = new_artifact_id()

        self.assertRegex(first, r"^\d{8}T\d{6}Z_[0-9a-f]{32}$")
        self.assertRegex(second, r"^\d{8}T\d{6}Z_[0-9a-f]{32}$")
        self.assertNotEqual(first, second)

    def test_publish_reuses_frozen_base_across_gcs_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_layout = PublicationLayout("gs://first/polyrepo/product")
            second_layout = PublicationLayout("gs://second/polyrepo/product")
            snapshot_path = Path(directory) / "snapshot"
            snapshot_path.mkdir()
            (snapshot_path / "base.tar.gz").write_bytes(b"base")
            (snapshot_path / "freeze_meta.json").write_text(
                '{"manifest_digest": "manifest-digest", '
                '"published_gcs_roots": ["gs://first/polyrepo/product"]}',
                encoding="utf-8",
            )
            state = SimpleNamespace(
                manifest_digest="manifest-digest",
                workspace=SimpleNamespace(remote_path=Path("workspace")),
                repositories={},
                sync=SimpleNamespace(snapshot_path=snapshot_path),
            )
            patch_uris = [
                first_layout.patch_uri("first"),
                second_layout.patch_uri("second"),
            ]

            with (
                patch("polyrepo.sync_repo.load_state", return_value=state),
                patch(
                    "polyrepo.sync_repo._compute_patch",
                    side_effect=patch_uris,
                ),
                patch("polyrepo.sync_repo.GsutilArtifactStore.upload") as upload,
            ):
                first = publish(Path(directory), first_layout.gcs_root)
                second = publish(Path(directory), second_layout.gcs_root)

            upload.assert_called_once_with(
                snapshot_path / "base.tar.gz",
                second_layout.frozen_base_uri,
            )
            self.assertEqual(first.base_uri, first_layout.frozen_base_uri)
            self.assertEqual(second.base_uri, second_layout.frozen_base_uri)
            self.assertEqual(
                json.loads(
                    (snapshot_path / "freeze_meta.json").read_text(
                        encoding="utf-8"
                    )
                )["published_gcs_roots"],
                [first_layout.gcs_root, second_layout.gcs_root],
            )

    def test_generator_help_requires_launch_target(self) -> None:
        output = io.StringIO()
        with patch.object(sys, "argv", ["generator", "--help"]):
            with contextlib.redirect_stdout(output):
                with self.assertRaisesRegex(SystemExit, "0"):
                    generator._parse_args()

        self.assertIn("Required for every invocation.", output.getvalue())

    def test_queue_affinity_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "enqueue",
                "hostname",
                "--node-prefix",
                "v6e-16-node",
                "--node-range",
                "1-3",
            ]
        )
        self.assertEqual(
            affinity_query(args),
            "node_id IN (1, 2, 3) AND node_prefix = 'v6e-16-node'",
        )

        args = parser.parse_args(
            ["reassign", "abc123", "--node-range", "1,3-5"]
        )
        self.assertEqual(args.command, "reassign")
        self.assertEqual(args.job_id, "abc123")
        self.assertEqual(affinity_query(args), "node_id IN (1, 3, 4, 5)")

        args = parser.parse_args(
            [
                "enqueue",
                "hostname",
                "--node-id",
                "1",
                "--node-range",
                "1-3",
            ]
        )
        with self.assertRaisesRegex(
            QueueCliError,
            "--node-id cannot be combined with --node-range",
        ):
            affinity_query(args)

    def test_run_script_uses_manifest_gcs_root(self) -> None:
        args = build_parser().parse_args(["run-script", "job.sh", "--node-id", "21"])
        client = Mock()
        client.post.return_value = {"job_id": "job-123"}
        uploaded = {
            "local_path": "/workspace/job.sh",
            "gcs_path": "gs://configured/scratch/oneoff/2026-07-21/hash_job.sh",
        }

        with patch(
            "tpu_dispatch_cli.queue_cli.resolve_default_gcs_root",
            return_value="gs://configured",
        ) as resolve_root:
            with patch(
                "tpu_dispatch_cli.queue_cli.upload_script_to_gcs",
                return_value=uploaded,
            ) as upload:
                result = run_script(client, args)

        resolve_root.assert_called_once_with()
        upload.assert_called_once_with("job.sh", "gs://configured/scratch/oneoff")
        self.assertFalse(hasattr(args, "gcs_base"))
        self.assertEqual(result["job_id"], "job-123")


if __name__ == "__main__":
    unittest.main()

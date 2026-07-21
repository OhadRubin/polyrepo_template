#!/usr/bin/env python3
"""
- description:
    Command-line client for the local tpu-dispatch queue API. This ports the
    useful deploy_tpu queue-cli surface to tpu-dispatch without importing
    deploy_tpu: queue CRUD, job lookup, cancellation, transition inspection,
    node/job status, and one-off script enqueueing.
- usage:
    # list jobs through the local dispatch daemon
    queue-cli list
    # enqueue a regular command for node 21
    queue-cli enqueue 'hostname' --node-id 21
    # enqueue a regular command for any v6e node 1 through 20
    queue-cli enqueue 'hostname' --node-range 1-20 --node-prefix v6e-16-node
    # move a queued or active job to any v6e node 1 through 20
    queue-cli reassign 76332eed --node-range 1-20 --node-prefix v6e-16-node
    # upload a script to GCS and enqueue a fetch-and-run command
    queue-cli run-script ./job.sh --node-id 21
    # inspect the worker.py tmux output for a node or assigned job
    queue-cli logs --node-id 21 --lines 200
    queue-cli logs --job-id 76332eed --lines 200
- user_story:
    content:
        As the operator, Ohad needs the queue to have the same practical CLI
        affordances it had in deploy_tpu, but pointed at tpu-dispatch's
        durable SQLite-backed HTTP API on port 8105. The CLI should make ad-hoc
        job launch, cancellation, inspection, and cleanup boring, while keeping
        target selection explicit enough that TPU shape guesses never decide
        where work runs.
    was_generated_via_skill: false
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


API_KEY_ENV = "DEPLOY_API_KEY"
SERVER_URL_ENV = "TPU_DISPATCH_SERVER_URL"
LIVE_VIEW_URL_ENV = "TPU_LIVE_VIEW_URL"
LOCAL_DISPATCH_URL = "http://localhost:8105"
LOCAL_LIVE_VIEW_URL = "http://localhost:8066"
ONEOFF_GCS_BASE = "gs://example-bucket/scratch/oneoff"
DEFAULT_N_WORKERS = 4
DEFAULT_PRIORITY = 1
DEFAULT_JOB_CLASS = "regular"
JOB_CLASSES = ("regular", "idle")
ACTIVE_STATUSES = {"assigned", "processing", "canceling"}
VALID_STATUSES = {"queued", "assigned", "processing", "canceling", "completed", "failed", "canceled"}


def is_wsl() -> bool:
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8")
    except OSError:
        return False
    return "microsoft" in release.lower() or "wsl" in release.lower()


def wsl_default_gateway() -> str | None:
    try:
        route = subprocess.check_output(
            ["ip", "route", "show", "default"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    parts = route.split()
    if "via" not in parts:
        return None
    via_idx = parts.index("via")
    if via_idx + 1 >= len(parts):
        return None
    return parts[via_idx + 1]


def tcp_connects(host: str, port: int, timeout: float) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
        except OSError:
            return False
    return True


def resolve_local_service_url(local_url: str, port: int) -> str:
    if tcp_connects("localhost", port, 0.4):
        return local_url
    if is_wsl():
        gateway = wsl_default_gateway()
        if gateway and tcp_connects(gateway, port, 0.4):
            return f"http://{gateway}:{port}"
    return local_url


class QueueCliError(Exception):
    """A user-facing queue operation failed."""


class TracebackArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("formatter_class", argparse.ArgumentDefaultsHelpFormatter)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:
        raise QueueCliError(f"{self.prog}: {message}\n{self.format_usage().strip()}")


@dataclass(frozen=True)
class DispatchClientConfig:
    base_url: str
    api_key: str

    @classmethod
    def resolve(cls, base_url: str | None) -> "DispatchClientConfig":
        resolved_url = (
            base_url
            or os.environ.get(SERVER_URL_ENV)
            or resolve_local_service_url(LOCAL_DISPATCH_URL, 8105)
        )
        if API_KEY_ENV not in os.environ:
            raise QueueCliError(f"{API_KEY_ENV} must be set")
        api_key = os.environ[API_KEY_ENV]
        if api_key.strip() == "":
            raise QueueCliError(f"{API_KEY_ENV} must be non-empty")
        return cls(base_url=resolved_url.rstrip("/"), api_key=api_key)


class DispatchQueueClient:
    def __init__(self, config: DispatchClientConfig):
        self._client = httpx.Client(
            base_url=config.base_url,
            headers={"X-API-Key": config.api_key},
            timeout=30.0,
        )

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("GET", path, params=params)

    def post(self, path: str, *, json_body: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("POST", path, json_body=json_body)

    def delete(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("DELETE", path, params=params)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, json=json_body, params=params)
        except httpx.HTTPError as exc:
            raise QueueCliError(f"{method} {path} failed: {type(exc).__name__}: {exc}") from exc
        if response.status_code >= 400:
            raise QueueCliError(f"{method} {path} failed: HTTP {response.status_code}: {response.text}")
        if response.text.strip() == "":
            raise QueueCliError(f"{method} {path} returned an empty response")
        payload = response.json()
        if not isinstance(payload, dict):
            raise QueueCliError(f"{method} {path} returned {type(payload).__name__}, expected object")
        return payload

    def close(self) -> None:
        self._client.close()


@dataclass(frozen=True)
class LiveViewClientConfig:
    base_url: str
    api_key: str

    @classmethod
    def resolve(cls, base_url: str | None) -> "LiveViewClientConfig":
        resolved_url = (
            base_url
            or os.environ.get(LIVE_VIEW_URL_ENV)
            or resolve_local_service_url(LOCAL_LIVE_VIEW_URL, 8066)
        )
        if API_KEY_ENV not in os.environ:
            raise QueueCliError(f"{API_KEY_ENV} must be set")
        api_key = os.environ[API_KEY_ENV]
        if api_key.strip() == "":
            raise QueueCliError(f"{API_KEY_ENV} must be non-empty")
        return cls(base_url=resolved_url.rstrip("/"), api_key=api_key)


class LiveViewClient:
    def __init__(self, config: LiveViewClientConfig):
        self._client = httpx.Client(
            base_url=config.base_url,
            headers={"X-API-Key": config.api_key},
            timeout=httpx.Timeout(30.0, read=120.0),
        )

    def get(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise QueueCliError(f"GET {path} failed: {type(exc).__name__}: {exc}") from exc
        if response.status_code >= 400:
            raise QueueCliError(f"GET {path} failed: HTTP {response.status_code}: {response.text}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise QueueCliError(f"GET {path} returned {type(payload).__name__}, expected object")
        return payload

    def close(self) -> None:
        self._client.close()


def emit_json(value: Any) -> None:
    print(json.dumps(value, sort_keys=True))


def sql_string_literal(value: str) -> str:
    if value == "":
        raise QueueCliError("SQL string literal cannot be empty")
    return "'" + value.replace("'", "''") + "'"


def node_range_query(spec: str) -> str:
    node_ids = sorted(parse_node_ids(spec))
    if len(node_ids) == 1:
        return f"node_id = {node_ids[0]}"
    return f"node_id IN ({', '.join(str(node_id) for node_id in node_ids)})"


def affinity_query(args: argparse.Namespace) -> str | None:
    generated_parts: list[str] = []
    if (getattr(args, "node_id", None) is not None
            and getattr(args, "node_range", None) is not None):
        raise QueueCliError("--node-id cannot be combined with --node-range")
    if getattr(args, "node_id", None) is not None:
        generated_parts.append(f"node_id = {args.node_id}")
    if getattr(args, "node_range", None) is not None:
        generated_parts.append(node_range_query(args.node_range))
    if getattr(args, "tpu_name", None) is not None:
        generated_parts.append(f"pod_name = {sql_string_literal(args.tpu_name)}")
    if getattr(args, "zone", None) is not None:
        generated_parts.append(f"zone = {sql_string_literal(args.zone)}")
    if getattr(args, "node_prefix", None) is not None:
        generated_parts.append(f"node_prefix = {sql_string_literal(args.node_prefix)}")

    if args.affinity is not None and generated_parts:
        raise QueueCliError(
            "--affinity cannot be combined with generated affinity flags")
    if args.affinity is not None:
        if args.affinity.strip() == "":
            raise QueueCliError("--affinity must be non-empty")
        return args.affinity
    if not generated_parts:
        return None
    return " AND ".join(generated_parts)


def require_positive_int(value: int, label: str) -> int:
    if value < 1:
        raise QueueCliError(f"{label} must be >= 1, got {value}")
    return value


def require_non_empty(value: str, label: str) -> str:
    if value.strip() == "":
        raise QueueCliError(f"{label} must be non-empty")
    return value


def enqueue_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_workers": require_positive_int(args.n_workers, "n_workers"),
        "command": require_non_empty(args.cmd, "command"),
        "priority": args.priority,
        "job_class": args.job_class,
        "pod_affinity_query": affinity_query(args),
    }


def enqueue_job(client: DispatchQueueClient, args: argparse.Namespace) -> dict[str, Any]:
    payload = enqueue_payload(args)
    result = client.post("/queue/enqueue", json_body=payload)
    if "job_id" not in result:
        raise QueueCliError(f"enqueue response missing job_id: {result!r}")
    return {"job_id": result["job_id"], "payload": payload}


def command_preview(command: str) -> str:
    if len(command) <= 70:
        return command
    return command[:67] + "..."


def print_jobs(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No jobs in queue.")
        return
    for row in rows:
        pod = row.get("assigned_to") or "-"
        job_class = row.get("job_class") or "?"
        print(
            f"[{row['status']:10}] {row['id'][:8]} | "
            f"pri={row['priority']:3} | class={job_class:7} | "
            f"n={row['n_workers']} | pod={pod:20} | "
            f"{command_preview(row['command'])}"
        )


def list_jobs(client: DispatchQueueClient, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = client.get("/queue/jobs")["jobs"]
    if not isinstance(rows, list):
        raise QueueCliError("/queue/jobs response field 'jobs' must be a list")
    if args.status is not None:
        rows = [row for row in rows if row["status"] == args.status]
    if args.filter_query is not None:
        rows = [
            row for row in rows
            if args.filter_query in (row.get("assigned_to") or "")
            or args.filter_query in row.get("command", "")
            or args.filter_query in (row.get("pod_affinity_query") or "")
        ]
    return rows


def show_job(client: DispatchQueueClient, job_id: str) -> dict[str, Any]:
    return client.get(f"/queue/jobs/{job_id}")


def transitions_for_job(client: DispatchQueueClient, job_id: str) -> dict[str, Any]:
    return client.get(f"/queue/jobs/{job_id}/transitions")


def worker_tmux_logs(
    live_view: LiveViewClient,
    args: argparse.Namespace,
    dispatch: DispatchQueueClient | None,
) -> dict[str, Any]:
    lines = require_positive_int(args.lines, "lines")
    if args.node_id is not None:
        return live_view.get(f"/api/nodes/{args.node_id}/worker-tmux", params={"lines": lines})
    if args.tpu_name is not None:
        return live_view.get(f"/api/tpus/{args.tpu_name}/worker-tmux", params={"lines": lines})
    if dispatch is None:
        raise QueueCliError("--job-id log lookup requires a dispatch client")
    job = show_job(dispatch, args.job_id)
    if job.get("assigned_node_id") is not None:
        return live_view.get(f"/api/nodes/{job['assigned_node_id']}/worker-tmux", params={"lines": lines})
    if job.get("assigned_to"):
        return live_view.get(f"/api/tpus/{job['assigned_to']}/worker-tmux", params={"lines": lines})
    raise QueueCliError(f"job {job['id']} is not assigned to a node or TPU")


def cancel_job(client: DispatchQueueClient, job_id: str) -> dict[str, Any]:
    result = client.post(f"/queue/jobs/{job_id}/cancel")
    if not ({"canceled", "canceling"} & set(result)):
        raise QueueCliError(f"cancel response missing canceled/canceling key: {result!r}")
    return result


def reassign_job(client: DispatchQueueClient, args: argparse.Namespace) -> dict[str, Any]:
    target_affinity = affinity_query(args)
    if target_affinity is None:
        raise QueueCliError("reassign requires an affinity target")
    payload = {"pod_affinity_query": target_affinity}
    return {
        "job": client.post(f"/queue/jobs/{args.job_id}/reassign",
                           json_body=payload),
        "payload": payload,
    }


def parse_node_ids(spec: str) -> set[int]:
    node_ids: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if part == "":
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            node_ids.update(range(int(start), int(end) + 1))
            continue
        node_ids.add(int(part))
    if not node_ids:
        raise QueueCliError(f"empty node id list: {spec!r}")
    return node_ids


def trailing_node_id(pod_name: str | None) -> int | None:
    if pod_name is None:
        return None
    suffix = pod_name.rsplit("-", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return None


def job_matches_node(row: dict[str, Any], node_ids: set[int]) -> bool:
    assigned_node_id = row.get("assigned_node_id")
    if assigned_node_id in node_ids:
        return True
    return trailing_node_id(row.get("assigned_to")) in node_ids


def stop_node_jobs(client: DispatchQueueClient, args: argparse.Namespace) -> dict[str, Any]:
    node_ids = parse_node_ids(args.node_ids)
    rows = client.get("/queue/jobs")["jobs"]
    targets = [
        row for row in rows
        if row["status"] in ACTIVE_STATUSES and job_matches_node(row, node_ids)
    ]
    canceled: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in targets:
        try:
            result = cancel_job(client, row["id"])
        except QueueCliError as exc:
            skipped.append({"job_id": row["id"], "assigned_to": row.get("assigned_to"), "error": str(exc)})
            continue
        canceled.append({"job_id": row["id"], "assigned_to": row.get("assigned_to"), "result": result})
    return {"node_ids": sorted(node_ids), "canceled": canceled, "skipped": skipped}


def purge_node_jobs(client: DispatchQueueClient, args: argparse.Namespace) -> dict[str, Any]:
    node_ids = parse_node_ids(args.node_ids)
    return client.post("/queue/purge-node",
                       json_body={"node_ids": sorted(node_ids)})


def purge_jobs(client: DispatchQueueClient, args: argparse.Namespace) -> dict[str, Any]:
    status = args.status
    if status == "all":
        status = None
    if status is None and not args.force:
        raise QueueCliError("refusing to purge all jobs without --force")
    params: dict[str, Any] = {}
    if status is not None:
        if status not in VALID_STATUSES:
            raise QueueCliError(f"invalid status {status!r}; expected one of {sorted(VALID_STATUSES)}")
        params["status"] = status
    return client.delete("/queue/jobs", params=params)


def upload_script_to_gcs(script_path: str, gcs_base: str) -> dict[str, str]:
    path = Path(script_path).expanduser().resolve()
    if not path.is_file():
        raise QueueCliError(f"script path is not a file: {path}")
    content = path.read_bytes()
    content_hash = hashlib.sha256(content).hexdigest()[:8]
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    gcs_path = f"{gcs_base.rstrip('/')}/{date}/{content_hash}_{path.stem}.sh"
    try:
        subprocess.run(["gsutil", "cp", str(path), gcs_path],
                       capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise QueueCliError(f"gsutil upload failed: {exc.stderr.strip()}") from exc
    return {"local_path": str(path), "gcs_path": gcs_path}


def run_script(client: DispatchQueueClient, args: argparse.Namespace) -> dict[str, Any]:
    target_affinity = affinity_query(args)
    uploaded = upload_script_to_gcs(args.script_path, args.gcs_base)
    command = f"gsutil cat {uploaded['gcs_path']} > /tmp/job.sh && bash /tmp/job.sh"
    payload = {
        "n_workers": require_positive_int(args.n_workers, "n_workers"),
        "command": command,
        "priority": args.priority,
        "job_class": args.job_class,
        "pod_affinity_query": target_affinity,
    }
    result = client.post("/queue/enqueue", json_body=payload)
    if "job_id" not in result:
        raise QueueCliError(f"enqueue response missing job_id: {result!r}")
    return {"job_id": result["job_id"], "upload": uploaded, "payload": payload}


def add_affinity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--affinity", help="SQL WHERE fragment over node_id/pod_name/zone/node_prefix")
    parser.add_argument("--node-id", type=int, help="Build affinity: node_id = N")
    parser.add_argument("--node-range", help="Build affinity: node_id IN (...) from 1-3 or 1,3-5")
    parser.add_argument("--tpu-name", help="Build affinity: pod_name = TPU")
    parser.add_argument("--zone", help="Build affinity: zone = ZONE")
    parser.add_argument("--node-prefix", help="Build affinity: node_prefix = PREFIX")


def build_parser() -> argparse.ArgumentParser:
    parser = TracebackArgumentParser(description="tpu-dispatch queue CLI")
    parser.add_argument("--server-url", help=f"Dispatch base URL; otherwise {SERVER_URL_ENV}, then {LOCAL_DISPATCH_URL}")
    parser.add_argument("--live-view-url", help=f"Live-view base URL for logs; otherwise {LIVE_VIEW_URL_ENV}, then {LOCAL_LIVE_VIEW_URL}")
    parser.add_argument("--json", dest="json_mode", action="store_true", help="Emit one JSON value on stdout")
    subparsers = parser.add_subparsers(dest="command")

    enqueue_parser = subparsers.add_parser("enqueue", help="Enqueue a command")
    enqueue_parser.add_argument("cmd")
    enqueue_parser.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS,
                                help="Number of TPU workers required")
    enqueue_parser.add_argument("--priority", type=int, default=DEFAULT_PRIORITY,
                                help="Queue priority; higher claims first")
    enqueue_parser.add_argument("--job-class", choices=JOB_CLASSES, default=DEFAULT_JOB_CLASS,
                                help="Dispatch class")
    add_affinity_arguments(enqueue_parser)

    run_script_parser = subparsers.add_parser("run-script", help="Upload a script to GCS and enqueue it")
    run_script_parser.add_argument("script_path")
    run_script_parser.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS,
                                   help="Number of TPU workers required")
    run_script_parser.add_argument("--priority", type=int, default=DEFAULT_PRIORITY,
                                   help="Queue priority; higher claims first")
    run_script_parser.add_argument("--job-class", choices=JOB_CLASSES, default=DEFAULT_JOB_CLASS,
                                   help="Dispatch class")
    run_script_parser.add_argument("--gcs-base", default=ONEOFF_GCS_BASE,
                                   help="GCS prefix for uploaded one-off scripts")
    add_affinity_arguments(run_script_parser)

    list_parser = subparsers.add_parser("list", help="List jobs")
    list_parser.add_argument("--filter-query")
    list_parser.add_argument("--status", choices=sorted(VALID_STATUSES))

    show_parser = subparsers.add_parser("show", help="Show one job by id or prefix")
    show_parser.add_argument("job_id")

    transitions_parser = subparsers.add_parser("transitions", help="Show one job's transition ledger")
    transitions_parser.add_argument("job_id")

    cancel_parser = subparsers.add_parser("cancel", help="Cancel one job by id or prefix")
    cancel_parser.add_argument("job_id")

    reassign_parser = subparsers.add_parser("reassign", help="Change one job's affinity and requeue it if active")
    reassign_parser.add_argument("job_id")
    add_affinity_arguments(reassign_parser)

    logs_parser = subparsers.add_parser("logs", help="Capture worker.py tmux output")
    logs_target = logs_parser.add_mutually_exclusive_group(required=True)
    logs_target.add_argument("--node-id", type=int)
    logs_target.add_argument("--tpu-name")
    logs_target.add_argument("--job-id")
    logs_parser.add_argument("--lines", type=int, required=True)

    stop_node_parser = subparsers.add_parser("stop-node", help="Cancel running jobs assigned to node ids")
    stop_node_parser.add_argument("node_ids")

    purge_node_parser = subparsers.add_parser("purge-node", help="Stop running jobs and cancel queued jobs stranded on node ids")
    purge_node_parser.add_argument("node_ids")

    purge_parser = subparsers.add_parser("purge", help="Delete jobs by status, or all with --force")
    purge_parser.add_argument("--status", choices=sorted(VALID_STATUSES | {"all"}))
    purge_parser.add_argument("--force", "-f", action="store_true")

    subparsers.add_parser("status", help="Show dispatch service status")
    subparsers.add_parser("nodes", help="Show dispatch node view")
    return parser


def print_mapping(value: dict[str, Any]) -> None:
    for key, item in value.items():
        print(f"{key}: {item}")


def run_command(client: DispatchQueueClient, args: argparse.Namespace) -> Any:
    if args.command == "enqueue":
        return enqueue_job(client, args)
    if args.command == "run-script":
        return run_script(client, args)
    if args.command == "list":
        return list_jobs(client, args)
    if args.command == "show":
        return show_job(client, args.job_id)
    if args.command == "transitions":
        return transitions_for_job(client, args.job_id)
    if args.command == "cancel":
        return cancel_job(client, args.job_id)
    if args.command == "reassign":
        return reassign_job(client, args)
    if args.command == "stop-node":
        return stop_node_jobs(client, args)
    if args.command == "purge-node":
        return purge_node_jobs(client, args)
    if args.command == "purge":
        return purge_jobs(client, args)
    if args.command == "status":
        return client.get("/status")
    if args.command == "nodes":
        return client.get("/nodes")
    raise QueueCliError("subcommand is required")


def run_logs_command(args: argparse.Namespace) -> dict[str, Any]:
    live_view = LiveViewClient(LiveViewClientConfig.resolve(args.live_view_url))
    dispatch: DispatchQueueClient | None = None
    if args.job_id is not None:
        dispatch = DispatchQueueClient(DispatchClientConfig.resolve(args.server_url))
    try:
        return worker_tmux_logs(live_view, args, dispatch)
    finally:
        live_view.close()
        if dispatch is not None:
            dispatch.close()


def print_human_result(args: argparse.Namespace, result: Any) -> None:
    if args.command == "enqueue":
        print(f"Enqueued job {result['job_id'][:8]}")
        print(json.dumps(result["payload"], indent=2, sort_keys=True))
        return
    if args.command == "run-script":
        print(f"Uploaded {result['upload']['local_path']} -> {result['upload']['gcs_path']}")
        print(f"Enqueued job {result['job_id'][:8]}")
        return
    if args.command == "list":
        print_jobs(result)
        return
    if args.command in {"show", "status", "nodes"}:
        print_mapping(result)
        return
    if args.command == "transitions":
        for row in result["transitions"]:
            print_mapping(row)
            print()
        return
    if args.command == "logs":
        for line in result["output"].splitlines():
            if line.strip():
                print(line)
        return
    if args.command == "cancel":
        status_key = "canceled" if "canceled" in result else "canceling"
        print(f"{status_key}: {result[status_key]}")
        return
    if args.command == "reassign":
        job = result["job"]
        print(f"Reassigned job {job['id'][:8]} -> {job['status']}")
        print(json.dumps(result["payload"], indent=2, sort_keys=True))
        return
    if args.command == "stop-node":
        print(f"Canceled {len(result['canceled'])} running job(s) for nodes {result['node_ids']}")
        for row in result["canceled"]:
            print(f"  {row['job_id'][:8]} {row['assigned_to']}")
        for row in result["skipped"]:
            print(f"  skip {row['job_id'][:8]} {row['assigned_to']}: {row['error']}")
        return
    if args.command == "purge-node":
        print(f"Nodes {result['node_ids']}: {len(result['canceling'])} running -> canceling, "
              f"{len(result['canceled'])} queued -> canceled")
        for row in result["canceling"]:
            print(f"  canceling {row['job_id'][:8]} {row['assigned_to']}")
        for row in result["canceled"]:
            print(f"  canceled  {row['job_id'][:8]} [{row['pod_affinity_query']}]")
        return
    if args.command == "purge":
        print(f"Deleted {result['deleted']} job(s).")
        return
    raise QueueCliError(f"no printer for command {args.command!r}")


def cli(argv: list[str]) -> int:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    if args.command == "logs":
        result = run_logs_command(args)
        if args.json_mode:
            emit_json(result)
            return 0
        print_human_result(args, result)
        return 0
    client = DispatchQueueClient(DispatchClientConfig.resolve(args.server_url))
    try:
        result = run_command(client, args)
    finally:
        client.close()
    if args.json_mode:
        emit_json(result)
        return 0
    print_human_result(args, result)
    return 0


def main() -> None:
    json_mode = "--json" in sys.argv[1:]
    try:
        raise SystemExit(cli(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception as exc:
        traceback.print_exc()
        if json_mode:
            emit_json({"error": str(exc)})
        else:
            print(exc, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

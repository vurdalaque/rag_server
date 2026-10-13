#!/usr/bin/env python3
"""CLI for /admin/retrieval operations (no psql/docker required).

Environment:
  RAG_ADMIN_TOKEN   — required
  RAG_BASE_URL      — default http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def _base_url() -> str:
    return os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _token() -> str:
    token = os.environ.get("RAG_ADMIN_TOKEN", "").strip()
    if not token:
        print("RAG_ADMIN_TOKEN is not set", file=sys.stderr)
        sys.exit(1)
    return token


def _request(
    method: str,
    path: str,
    *,
    query: dict[str, str | int | bool] | None = None,
    body: dict[str, Any] | None = None,
) -> Any:
    url = f"{_base_url()}{path}"
    if query:
        params = urllib.parse.urlencode(
            {
                key: str(value).lower() if isinstance(value, bool) else value
                for key, value in query.items()
            }
        )
        url = f"{url}?{params}"
    data = None
    headers = {"Authorization": f"Bearer {_token()}"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = resp.read().decode("utf-8")
            if not payload:
                return None
            return json.loads(payload)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        print(f"HTTP {error.code}: {detail}", file=sys.stderr)
        sys.exit(1)


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _cmd_status(args: argparse.Namespace) -> None:
    data = _request(
        "GET",
        "/admin/retrieval/stats",
        query={"start": args.start, "end": args.end},
    )
    if args.json:
        _print_json(data)
        return
    coverage = data.get("coverage", {})
    ops = data.get("evaluation_ops", {})
    print(f"evaluation_version: {coverage.get('evaluation_version')}")
    print(
        f"coverage: {coverage.get('evaluated_events')}/{coverage.get('total_events')} "
        f"({coverage.get('coverage_ratio')}) pending_jobs={coverage.get('pending_jobs')}"
    )
    print(
        f"queue (all versions): depth={ops.get('queue_depth')} "
        f"failed={ops.get('failed_jobs')} dead={ops.get('dead_jobs')}"
    )


def _cmd_ops(args: argparse.Namespace) -> None:
    data = _request("GET", "/admin/retrieval/ops")
    if args.json:
        _print_json(data)
        return
    print(f"evaluation_version: {data.get('evaluation_version')}")
    print(f"missing_completed_evaluations: {data.get('missing_completed_evaluations')}")
    print(f"requeue_retryable_count: {data.get('requeue_retryable_count')}")
    print("jobs_by_status:")
    for status, count in sorted((data.get("jobs_by_status") or {}).items()):
        print(f"  {status}: {count}")
    print("failed_reasons:")
    for row in data.get("failed_reasons") or []:
        print(f"  {row.get('reason')}: {row.get('count')}")


def _cmd_backlog(args: argparse.Namespace) -> None:
    query: dict[str, str | int | bool] = {
        "limit": args.limit,
        "only_missing_completed": not args.all_events,
    }
    if args.cursor_created_at:
        query["cursor_created_at"] = args.cursor_created_at
    if args.cursor_event_id:
        query["cursor_event_id"] = args.cursor_event_id
    data = _request("GET", "/admin/retrieval/backlog", query=query)
    if args.json:
        _print_json(data)
        return
    print(f"evaluation_version: {data.get('evaluation_version')} count={data.get('count')}")
    for row in data.get("items") or []:
        err = (row.get("last_error") or "")[:80].replace("\n", " ")
        print(
            f"{row.get('event_id')}  {row.get('event_created_at')}  "
            f"job={row.get('job_status') or '-'}  retryable={row.get('retryable')}  "
            f"reason={row.get('failure_reason') or '-'}  err={err}"
        )
    if data.get("next_cursor"):
        print("next_cursor:", json.dumps(data["next_cursor"]))


def _cmd_bulk(args: argparse.Namespace) -> None:
    body: dict[str, Any] = {
        "start": args.start,
        "end": args.end,
        "limit": args.limit,
    }
    if args.cursor_created_at:
        body["cursor_created_at"] = args.cursor_created_at
    if args.cursor_event_id:
        body["cursor_event_id"] = args.cursor_event_id
    data = _request("POST", "/admin/retrieval/evaluate/bulk", body=body)
    _print_json(data)


def _cmd_requeue_failed(args: argparse.Namespace) -> None:
    body: dict[str, Any] = {
        "skip_data_errors": not args.include_data_errors,
        "include_dead": not args.failed_only,
    }
    if args.evaluation_version:
        body["evaluation_version"] = args.evaluation_version
    data = _request("POST", "/admin/retrieval/jobs/requeue-failed", body=body)
    _print_json(data)


def _cmd_requeue_one(args: argparse.Namespace) -> None:
    path = f"/admin/retrieval/evaluate/{args.event_id}"
    data = _request(
        "POST",
        path,
        body={"event_created_at": args.event_created_at},
    )
    _print_json(data)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieval observability admin CLI")
    parser.add_argument("--json", action="store_true", help="Print raw JSON responses")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Coverage + global queue stats")
    status.add_argument("--start", default="2026-01-01T00:00:00Z")
    status.add_argument("--end", default="2026-12-31T23:59:59Z")
    status.set_defaults(func=_cmd_status)

    sub.add_parser("ops", help="Jobs by status, failure reasons, requeue hints").set_defaults(
        func=_cmd_ops
    )

    backlog = sub.add_parser("backlog", help="Events missing completed evaluation")
    backlog.add_argument("--limit", type=int, default=20)
    backlog.add_argument("--all-events", action="store_true", help="Include events with completed eval")
    backlog.add_argument("--cursor-created-at", default="")
    backlog.add_argument("--cursor-event-id", default="")
    backlog.set_defaults(func=_cmd_backlog)

    bulk = sub.add_parser("bulk", help="Enqueue evaluation jobs for events in time range")
    bulk.add_argument("--start", default="2026-01-01T00:00:00Z")
    bulk.add_argument("--end", default="2026-12-31T23:59:59Z")
    bulk.add_argument("--limit", type=int, default=500)
    bulk.add_argument("--cursor-created-at", default="")
    bulk.add_argument("--cursor-event-id", default="")
    bulk.set_defaults(func=_cmd_bulk)

    requeue_failed = sub.add_parser(
        "requeue-failed",
        help="Reset failed/dead jobs to pending (skips empty_final_top_k by default)",
    )
    requeue_failed.add_argument("--evaluation-version", default="")
    requeue_failed.add_argument(
        "--include-data-errors",
        action="store_true",
        help="Also requeue empty_final_top_k / missing_chunk_text failures",
    )
    requeue_failed.add_argument(
        "--failed-only",
        action="store_true",
        help="Do not reset dead jobs",
    )
    requeue_failed.set_defaults(func=_cmd_requeue_failed)

    requeue_one = sub.add_parser("requeue", help="Re-enqueue a single event (reset_terminal)")
    requeue_one.add_argument("event_id")
    requeue_one.add_argument("event_created_at")
    requeue_one.set_defaults(func=_cmd_requeue_one)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

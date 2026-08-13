from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

from ..models import JobPlan, JobResult


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


@dataclass(slots=True)
class CliContext:
    json_output: bool = False
    quiet: bool = False
    verbose: bool = False
    stdout: TextIO = field(default_factory=lambda: sys.stdout)
    stderr: TextIO = field(default_factory=lambda: sys.stderr)

    def log(self, message: str) -> None:
        if not self.quiet and not self.json_output:
            print(message, file=self.stderr)

    def error(self, message: str) -> None:
        if self.json_output:
            self.emit({"status": "error", "message": message})
        else:
            print(f"error: {message}", file=self.stderr)

    def emit(self, value: Any) -> None:
        if self.json_output:
            print(json.dumps(_json_value(value), ensure_ascii=False, indent=2), file=self.stdout)
        elif isinstance(value, str):
            print(value, file=self.stdout)
        else:
            print(json.dumps(_json_value(value), ensure_ascii=False, indent=2), file=self.stdout)

    def emit_plan(self, plan: JobPlan, *, max_api_attempts: int | None = None) -> None:
        maximum_attempts = plan.api_requests if max_api_attempts is None else max_api_attempts
        payload = {
            "status": "conflict" if plan.conflicts else "planned",
            "dry_run": plan.dry_run,
            "api_requests": plan.api_requests,
            "max_api_attempts": maximum_attempts,
            "provider": plan.provider,
            "sources": [str(path) for path in plan.sources],
            "artifacts": [
                {"source": str(item.source), "target": str(item.target), "format": item.format.value}
                for item in plan.artifacts
            ],
            "conflicts": [
                {
                    "target": str(item.target),
                    "sources": [str(path) for path in item.sources],
                    "reason": item.reason,
                }
                for item in plan.conflicts
            ],
        }
        if self.json_output:
            self.emit(payload)
            return
        self.log(
            f"Plan: {len(plan.sources)} source(s), {len(plan.artifacts)} artifact(s), "
            f"{plan.api_requests} API request(s), up to {maximum_attempts} attempt(s)"
        )
        for artifact in plan.artifacts:
            self.log(f"  {artifact.format.value}: {artifact.source.name} -> {artifact.target}")
        for conflict in plan.conflicts:
            self.log(f"  CONFLICT {conflict.target}: {conflict.reason}")

    def emit_result(self, result: JobResult) -> None:
        payload = {
            "status": "failed" if result.failed else "ok",
            "written": result.written,
            "skipped": result.skipped,
            "failed": result.failed,
            "artifacts": [
                {
                    "source": str(item.artifact.source),
                    "target": str(item.artifact.target),
                    "format": item.artifact.format.value,
                    "status": item.status.value,
                    "message": item.message,
                }
                for item in result.artifacts
            ],
        }
        if self.json_output:
            self.emit(payload)
            return
        for item in result.artifacts:
            suffix = f" ({item.message})" if item.message else ""
            self.log(f"{item.status.value.upper():7} {item.artifact.target}{suffix}")
        self.log(f"Done: {result.written} written, {result.skipped} skipped, {result.failed} failed")

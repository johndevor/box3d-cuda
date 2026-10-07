from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "factory-os.physics-benchmark/v1"


@dataclass(frozen=True)
class CapabilitySet:
    rigid_body_integration: bool
    static_plane_contacts: bool
    dynamic_contacts: bool
    articulated_joints: bool
    continuous_collision: bool
    ray_queries: bool
    camera_rendering: bool
    robot_manipulation: bool

    def covers(self, required: "CapabilitySet") -> bool:
        return all(not expected or getattr(self, key) for key, expected in asdict(required).items())


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    backend_version: str
    workload: str
    contract_id: str
    device: str
    worlds: int
    bodies_per_world: int
    steps: int
    duration_seconds: float
    capabilities: CapabilitySet
    correctness: Mapping[str, Any]
    peak_memory_bytes: int | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if min(self.worlds, self.bodies_per_world, self.steps) <= 0:
            raise ValueError("worlds, bodies_per_world and steps must be positive")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be finite and positive")
        if self.correctness.get("passed") is not True:
            raise ValueError("performance evidence is invalid until correctness passes")

    @property
    def world_steps_per_second(self) -> float:
        return self.worlds * self.steps / self.duration_seconds

    @property
    def body_steps_per_second(self) -> float:
        return self.world_steps_per_second * self.bodies_per_world

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **asdict(self),
            "capabilities": asdict(self.capabilities),
            "world_steps_per_second": self.world_steps_per_second,
            "body_steps_per_second": self.body_steps_per_second,
        }


def write_result(path: Path, result: BenchmarkResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported benchmark schema in {path}")
    if payload.get("correctness", {}).get("passed") is not True:
        raise ValueError(f"benchmark did not pass correctness: {path}")
    return payload

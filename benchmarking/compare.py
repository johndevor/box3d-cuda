from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from .contract import load_result


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def compare(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    if len(rows) < 2:
        raise ValueError("comparison needs at least two benchmark results")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["contract_id"], []).append(row)
    speedups: list[dict[str, Any]] = []
    for contract_id, matches in groups.items():
        if len(matches) < 2:
            continue
        baseline = matches[0]
        for candidate in matches[1:]:
            speedups.append(
                {
                    "contract_id": contract_id,
                    "baseline": baseline["backend"],
                    "candidate": candidate["backend"],
                    "world_step_speedup": candidate["world_steps_per_second"] / baseline["world_steps_per_second"],
                }
            )
    return {
        "schema_version": "factory-os.physics-comparison/v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "results": rows,
        "speedups": speedups,
        "parity_warning": None if speedups else "No shared contract_id; a speedup claim would be invalid.",
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Physics backend comparison",
        "",
        "| Backend | Workload | Contract | Worlds | Bodies/world | World steps/s | Body steps/s | Contacts | Joints | Cameras |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in report["results"]:
        caps = row["capabilities"]
        lines.append(
            "| {backend} | {workload} | `{contract}` | {worlds:,} | {bodies:,} | {world_s:,.0f} | {body_s:,.0f} | {contacts} | {joints} | {cameras} |".format(
                backend=row["backend"],
                workload=row["workload"],
                contract=row["contract_id"],
                worlds=row["worlds"],
                bodies=row["bodies_per_world"],
                world_s=row["world_steps_per_second"],
                body_s=row["body_steps_per_second"],
                contacts=_yes(row["capabilities"]["dynamic_contacts"]),
                joints=_yes(caps["articulated_joints"]),
                cameras=_yes(caps["camera_rendering"]),
            )
        )
    lines.extend(["", "## Valid comparisons", ""])
    if report["speedups"]:
        for speedup in report["speedups"]:
            lines.append(
                f"- `{speedup['contract_id']}`: {speedup['candidate']} is {speedup['world_step_speedup']:.2f}× the world-step throughput of {speedup['baseline']}."
            )
    else:
        lines.append(f"- {report['parity_warning']}")
    lines.extend(
        [
            "",
            "A body-step is not treated as a robot-control step. Capability columns and contract IDs are part of the result, so an incomplete kernel cannot win by doing less work.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = compare(load_result(path) for path in args.results)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": "passed", "results": len(report["results"]), "valid_speedups": len(report["speedups"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

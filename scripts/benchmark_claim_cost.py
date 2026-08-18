"""Decompose the safe-vs-naive throughput gap into its two separate causes.

A single "safe mode is 7x slower" number invites the obvious question: slower
because of *what*? Two different things are bundled in there, and they have very
different standing:

  1. Serializable claiming -- one locking read plus one write in a transaction
     that retries on 40001. This is correctness, and it is not optional.
  2. Contention detection -- one extra unlocked read per claim, so the loser of
     a race can identify who beat it and write a conflict_log row. This is
     observability, and it *is* optional.

Run: .venv/Scripts/python scripts/benchmark_claim_cost.py
Writes: docs/stress-cost-breakdown.json
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from quorum.config import get_settings
from quorum.logging import configure_logging
from quorum.stress import run_stress
from quorum.workspace import seed_workspace

AGENTS = 8
ITERATIONS = 60
REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    settings = get_settings()
    configure_logging("WARNING", "console")

    spec = json.loads((REPO_ROOT / "tasks" / "requests-to-httpx.json").read_text())
    spec["repo"] = str(REPO_ROOT / spec["repo"])

    seeded = seed_workspace(
        name=f"cost-{uuid.uuid4().hex[:8]}",
        task_spec=spec,
        mode="safe",
        settings=settings,
    )

    configurations = [
        ("naive", {"mode": "naive", "barrier": False}),
        ("safe_no_detection", {"mode": "safe", "detect_contention": False}),
        ("safe_with_detection", {"mode": "safe", "detect_contention": True}),
    ]

    results: dict[str, dict] = {}
    for label, kwargs in configurations:
        print(f"running {label} ...", flush=True)
        report = run_stress(
            seeded.workspace_id,
            agents=AGENTS,
            iterations=ITERATIONS,
            settings=settings,
            **kwargs,
        )
        results[label] = report.to_dict()
        print(f"  {report.summary()}", flush=True)

    naive = results["naive"]["claims_per_second"]
    bare = results["safe_no_detection"]["claims_per_second"]
    full = results["safe_with_detection"]["claims_per_second"]

    breakdown = {
        "agents": AGENTS,
        "iterations": ITERATIONS,
        "units": results["naive"]["units"],
        "claims_per_second": {
            "naive_unsafe": naive,
            "safe_without_detection": bare,
            "safe_with_detection": full,
        },
        "slowdown_vs_naive": {
            "correctness_only": round(naive / bare, 2) if bare else None,
            "correctness_plus_observability": round(naive / full, 2) if full else None,
        },
        "observability_share": {
            "extra_slowdown_from_detection": round(bare / full, 2) if full else None,
            "note": (
                "Detection is one extra unlocked read per claim. It buys the "
                "conflict feed and costs nothing in correctness, so it can be "
                "sampled or disabled under load via QUORUM_CONFLICT_DETECTION."
            ),
        },
        "correctness": {
            "naive_duplicate_claims": results["naive"]["duplicate_claims"],
            "safe_without_detection_duplicate_claims": results["safe_no_detection"][
                "duplicate_claims"
            ],
            "safe_with_detection_duplicate_claims": results["safe_with_detection"][
                "duplicate_claims"
            ],
        },
        "conflicts_logged": {
            "naive": results["naive"]["conflicts_logged"],
            "safe_without_detection": results["safe_no_detection"]["conflicts_logged"],
            "safe_with_detection": results["safe_with_detection"]["conflicts_logged"],
        },
        "runs": results,
    }

    out = REPO_ROOT / "docs" / "stress-cost-breakdown.json"
    out.write_text(json.dumps(breakdown, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print(json.dumps(breakdown["slowdown_vs_naive"], indent=2))
    print(json.dumps(breakdown["observability_share"], indent=2))


if __name__ == "__main__":
    main()

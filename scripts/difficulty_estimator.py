#!/usr/bin/env python
"""
Difficulty Estimator - quick difficulty diagnostics for episode JSONL.

Usage:
  uv run python scripts/difficulty_estimator.py --episodes data/episodes/episodes.jsonl
  uv run python scripts/difficulty_estimator.py --episodes data/episodes/
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_episodes(path: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    else:
        files = [path]
    for jsonl_path in files:
        with open(jsonl_path) as f:
            for line in f:
                try:
                    ep = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ep["_file"] = jsonl_path.name
                episodes.append(ep)
    return episodes


def _entropy(counts: Counter) -> tuple[float, float]:
    total = sum(counts.values())
    if total <= 1:
        return 0.0, 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    k = len(counts)
    ent_norm = ent / math.log2(k) if k > 1 else 0.0
    return ent, ent_norm


def episode_stats(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(episodes)
    if total == 0:
        return {}

    verified = sum(1 for ep in episodes if ep.get("verified", False))

    turn_counts = []
    hooks_total = 0
    hooks_turns = 0
    corrections = 0
    gold_fail = 0

    consistency_total = 0
    consistency_success = 0
    majority_confs = []
    entropies = []
    entropies_norm = []

    for ep in episodes:
        gold = ep.get("gold_trace", {}) or {}
        turns = gold.get("turns", []) or []
        turn_counts.append(len(turns))

        for t in turns:
            exec_result = t.get("execution", {}) or {}
            hooks = exec_result.get("hooks", []) or []
            hooks_total += len(hooks)
            hooks_turns += 1
            if t.get("correction"):
                corrections += 1

        if not gold.get("success", False):
            gold_fail += 1

        consistency = ep.get("consistency_traces", []) or []
        if consistency:
            consistency_total += len(consistency)
            successes = [
                t for t in consistency if t.get("success", False)
            ]
            consistency_success += len(successes)

            hashes = []
            for t in successes:
                h = t.get("final_answer_hash")
                if h is None:
                    # Fall back to final_answer string for clustering
                    h = json.dumps(t.get("final_answer", None), sort_keys=True, default=str)
                hashes.append(h)

            if hashes:
                counts = Counter(hashes)
                majority_conf = max(counts.values()) / max(1, len(hashes))
                majority_confs.append(majority_conf)
                ent, ent_norm = _entropy(counts)
                entropies.append(ent)
                entropies_norm.append(ent_norm)

    avg_turns = sum(turn_counts) / max(1, len(turn_counts))
    pct_single_turn = sum(1 for t in turn_counts if t <= 1) / max(1, len(turn_counts))
    avg_hooks_per_turn = hooks_total / max(1, hooks_turns)
    correction_rate = corrections / max(1, hooks_turns)
    gold_fail_rate = gold_fail / max(1, total)

    consistency_success_rate = (
        consistency_success / max(1, consistency_total)
        if consistency_total > 0
        else None
    )
    avg_majority_conf = (
        sum(majority_confs) / len(majority_confs) if majority_confs else None
    )
    avg_entropy = sum(entropies) / len(entropies) if entropies else None
    avg_entropy_norm = (
        sum(entropies_norm) / len(entropies_norm) if entropies_norm else None
    )

    # Heuristic difficulty index (0-100). Higher = harder.
    # This is intentionally simple; treat as a directional signal only.
    verified_rate = verified / max(1, total)
    disagreement = (1 - avg_majority_conf) if avg_majority_conf is not None else 0.0
    turn_factor = min(avg_turns / 5.0, 1.0)
    difficulty_index = 100 * (
        0.45 * (1 - verified_rate)
        + 0.25 * disagreement
        + 0.20 * turn_factor
        + 0.10 * correction_rate
    )

    return {
        "total": total,
        "verified": verified,
        "verified_rate": verified_rate,
        "avg_turns": avg_turns,
        "pct_single_turn": pct_single_turn,
        "avg_hooks_per_turn": avg_hooks_per_turn,
        "correction_rate": correction_rate,
        "gold_fail_rate": gold_fail_rate,
        "consistency_success_rate": consistency_success_rate,
        "avg_majority_conf": avg_majority_conf,
        "avg_entropy": avg_entropy,
        "avg_entropy_norm": avg_entropy_norm,
        "difficulty_index": difficulty_index,
    }


def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if pct:
            return f"{v*100:.1f}%"
        return f"{v:.3f}"
    return str(v)


def print_stats(title: str, stats: dict[str, Any]) -> None:
    if not stats:
        print(f"\n[{title}] No episodes")
        return
    print(f"\n[{title}]")
    print(f"  total: {stats['total']}")
    print(f"  verified: {stats['verified']} ({_fmt(stats['verified_rate'], pct=True)})")
    print(f"  avg_turns: {_fmt(stats['avg_turns'])} | single_turn: {_fmt(stats['pct_single_turn'], pct=True)}")
    print(f"  avg_hooks/turn: {_fmt(stats['avg_hooks_per_turn'])} | correction_rate: {_fmt(stats['correction_rate'], pct=True)}")
    print(f"  gold_fail_rate: {_fmt(stats['gold_fail_rate'], pct=True)}")
    print(f"  consistency_success_rate: {_fmt(stats['consistency_success_rate'], pct=True)}")
    print(f"  avg_majority_conf: {_fmt(stats['avg_majority_conf'])} | avg_entropy_norm: {_fmt(stats['avg_entropy_norm'])}")
    print(f"  difficulty_index: {_fmt(stats['difficulty_index'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate dataset difficulty from episode JSONL.")
    parser.add_argument(
        "--episodes",
        type=str,
        required=True,
        help="Path to episodes.jsonl or a directory containing JSONL files",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="difficulty",
        help="Comma-separated grouping keys: difficulty,source,file",
    )
    args = parser.parse_args()

    path = Path(args.episodes)
    if not path.exists():
        print(f"Path not found: {path}")
        return 1

    episodes = load_episodes(path)
    if not episodes:
        print("No episodes found.")
        return 1

    print("DIFFICULTY ESTIMATOR")
    print(f"Loaded {len(episodes)} episodes from {path}")

    print_stats("overall", episode_stats(episodes))

    group_keys = [k.strip() for k in args.group_by.split(",") if k.strip()]
    for key in group_keys:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if key == "difficulty":
            for ep in episodes:
                diff = (ep.get("question", {}) or {}).get("difficulty", "UNKNOWN")
                groups[str(diff)] += [ep]
        elif key == "source":
            for ep in episodes:
                src = ep.get("source", "unknown")
                groups[str(src)] += [ep]
        elif key == "file":
            for ep in episodes:
                groups[str(ep.get("_file", "unknown"))] += [ep]
        else:
            print(f"\n[skip] unknown group key: {key}")
            continue

        for name in sorted(groups.keys()):
            print_stats(f"{key}={name}", episode_stats(groups[name]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

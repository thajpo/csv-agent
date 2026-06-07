#!/usr/bin/env python3
"""Summarize Prime-RL rollout JSONL into small repo artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from collections.abc import Sequence
from typing import Any


@dataclass(frozen=True)
class Point:
    step: int
    split: str
    env: str
    mean_reward: float
    n: int


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _find_reward(row: dict[str, Any]) -> float | None:
    for key in ("reward", "score"):
        reward = _numeric(row.get(key))
        if reward is not None:
            return reward

    raw = row.get("raw")
    if isinstance(raw, dict):
        for key in ("reward", "score"):
            reward = _numeric(raw.get(key))
            if reward is not None:
                return reward

    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        for key in ("reward", "reward_mean", "mean_reward"):
            reward = _numeric(metrics.get(key))
            if reward is not None:
                return reward

    return None


def _step_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.search(r"step[_-]?(\d+)", part)
        if match:
            return int(match.group(1))
    return -1


def _split_env_from_path(path: Path) -> tuple[str, str]:
    name = path.name
    if name == "train_rollouts.jsonl":
        return "train", "all"
    match = re.match(r"eval_rollouts_(.+)\.jsonl$", name)
    if match:
        return "eval", match.group(1)
    return "unknown", "unknown"


def collect_points(run_dir: Path) -> list[Point]:
    points: list[Point] = []
    for path in sorted(run_dir.rglob("*rollouts*.jsonl")):
        rewards = [
            reward
            for row in _jsonl_rows(path)
            if (reward := _find_reward(row)) is not None
        ]
        if not rewards:
            continue
        split, env = _split_env_from_path(path)
        points.append(
            Point(
                step=_step_from_path(path),
                split=split,
                env=env,
                mean_reward=mean(rewards),
                n=len(rewards),
            )
        )
    return sorted(points, key=lambda point: (point.step, point.split, point.env))


def write_json(points: list[Point], output_path: Path) -> None:
    output_path.write_text(
        json.dumps([point.__dict__ for point in points], indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(points: list[Point], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "split", "env", "mean_reward", "n"],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(point.__dict__)


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def write_svg(points: list[Point], output_path: Path, title: str) -> None:
    width = 760
    height = 420
    pad_left = 64
    pad_right = 28
    pad_top = 48
    pad_bottom = 58
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    if not points:
        output_path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="24" y="48" font-family="sans-serif" font-size="20">{title}</text>'
            '<text x="24" y="88" font-family="sans-serif" font-size="14">No reward points found.</text>'
            "</svg>\n",
            encoding="utf-8",
        )
        return

    steps = [point.step for point in points if point.step >= 0]
    min_step = min(steps) if steps else 0
    max_step = max(steps) if steps else 1
    if min_step == max_step:
        max_step = min_step + 1

    rewards = [point.mean_reward for point in points]
    min_reward = min(0.0, min(rewards))
    max_reward = max(1.0, max(rewards))
    if min_reward == max_reward:
        max_reward = min_reward + 1

    def scale_x(step: int) -> float:
        step = max(step, min_step)
        return pad_left + ((step - min_step) / (max_step - min_step)) * plot_w

    def scale_y(reward: float) -> float:
        return pad_top + (1 - ((reward - min_reward) / (max_reward - min_reward))) * plot_h

    grouped: dict[tuple[str, str], list[Point]] = {}
    for point in points:
        grouped.setdefault((point.split, point.env), []).append(point)

    palette = {
        "train": "#2166ac",
        "eval": "#b2182b",
        "unknown": "#4d4d4d",
    }

    series_svg = []
    legend_svg = []
    legend_y = pad_top
    for index, ((split, env), series) in enumerate(sorted(grouped.items())):
        color = palette.get(split, "#4d4d4d")
        coords = [(scale_x(point.step), scale_y(point.mean_reward)) for point in series]
        if len(coords) == 1:
            x, y = coords[0]
            series_svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" />')
        else:
            series_svg.append(
                f'<polyline points="{_polyline(coords)}" fill="none" stroke="{color}" '
                'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />'
            )
        label = f"{split}/{env}"
        legend_svg.append(
            f'<line x1="{width - 176}" y1="{legend_y + index * 22}" '
            f'x2="{width - 148}" y2="{legend_y + index * 22}" stroke="{color}" stroke-width="3" />'
            f'<text x="{width - 140}" y="{legend_y + index * 22 + 5}" '
            'font-family="sans-serif" font-size="13" fill="#222">'
            f"{label}</text>"
        )

    x0 = pad_left
    y0 = pad_top + plot_h
    grid_y = [
        (min_reward + (max_reward - min_reward) * tick / 4)
        for tick in range(5)
    ]
    grid_svg = []
    for reward in grid_y:
        y = scale_y(reward)
        grid_svg.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" '
            'stroke="#d8d8d8" stroke-width="1" />'
            f'<text x="{pad_left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            'font-family="sans-serif" font-size="12" fill="#555">'
            f"{reward:.2f}</text>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{pad_left}" y="28" font-family="sans-serif" font-size="20" font-weight="700" fill="#111">{title}</text>
  {''.join(grid_svg)}
  <line x1="{x0}" y1="{pad_top}" x2="{x0}" y2="{y0}" stroke="#222" stroke-width="1.2" />
  <line x1="{x0}" y1="{y0}" x2="{width - pad_right}" y2="{y0}" stroke="#222" stroke-width="1.2" />
  <text x="{pad_left + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#333">Prime-RL step</text>
  <text x="18" y="{pad_top + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {pad_top + plot_h / 2:.1f})" font-family="sans-serif" font-size="13" fill="#333">mean reward</text>
  <text x="{pad_left}" y="{y0 + 22}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#555">{min_step}</text>
  <text x="{width - pad_right}" y="{y0 + 22}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#555">{max_step}</text>
  {''.join(series_svg)}
  {''.join(legend_svg)}
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def write_readme(points: list[Point], output_path: Path, title: str) -> None:
    best = max(points, key=lambda point: point.mean_reward) if points else None
    best_line = (
        f"- Best mean reward: `{best.mean_reward:.4f}` at step `{best.step}` ({best.split}/{best.env})"
        if best
        else "- Best mean reward: unavailable"
    )
    output_path.write_text(
        f"""# {title}

![Reward curve](reward_curve.svg)

{best_line}
- Points summarized: `{len(points)}`

Generated from Prime-RL rollout JSONL files.
""",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Directory for small summary artifacts. Defaults to artifacts/prime_rl/<run-dir-name>.",
    )
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    artifact_dir = (
        args.artifact_dir.resolve()
        if args.artifact_dir
        else Path("artifacts") / "prime_rl" / run_dir.name
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    title = args.title or f"Prime-RL Run: {run_dir.name}"

    points = collect_points(run_dir)
    write_json(points, artifact_dir / "metrics.json")
    write_csv(points, artifact_dir / "metrics.csv")
    write_svg(points, artifact_dir / "reward_curve.svg", title)
    write_readme(points, artifact_dir / "README.md", title)

    print(f"Wrote {len(points)} points to {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

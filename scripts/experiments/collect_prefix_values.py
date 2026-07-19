"""Collect a small verifier-grounded value estimate for one turn boundary.

This command intentionally defaults to one episode and caps continuation
branches. It reports the planned rollout count before starting; each rollout
can require multiple turn-level model requests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from csv_spec import ContinuationPolicy, EpisodeJSONL

from src.value import (
    build_trajectory_prefix,
    collect_prefix_value,
    run_initial_model_trace,
)
from src.core.model import API_MAX_RETRIES

MAX_CONTINUATIONS = 16
MAX_EPISODES = 8
MAX_TURNS = 20
MAX_PROVIDER_REQUESTS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate future success from a replayed CSV-agent prefix."
    )
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument(
        "--csv",
        type=Path,
        help="Local CSV override for a one-dataset canary.",
    )
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--turn-count", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--continuations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--float-tolerance", type=float, default=0.1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.episodes.is_file():
        raise ValueError(f"episodes file does not exist: {args.episodes}")
    if args.csv is not None and not args.csv.is_file():
        raise ValueError(f"CSV override does not exist: {args.csv}")
    if not 1 <= args.max_episodes <= MAX_EPISODES:
        raise ValueError(f"max-episodes must be between 1 and {MAX_EPISODES}")
    if not 1 <= args.max_turns <= MAX_TURNS:
        raise ValueError(f"max-turns must be between 1 and {MAX_TURNS}")
    if args.turn_count < 0 or args.turn_count >= args.max_turns:
        raise ValueError("turn-count must be between zero and max-turns - 1")
    if not 1 <= args.continuations <= MAX_CONTINUATIONS:
        raise ValueError(f"continuations must be between 1 and {MAX_CONTINUATIONS}")
    request_limit = maximum_provider_requests(args)
    if request_limit > MAX_PROVIDER_REQUESTS:
        raise ValueError(
            f"configuration permits up to {request_limit} provider requests; "
            f"the canary limit is {MAX_PROVIDER_REQUESTS}"
        )


def maximum_provider_requests(args: argparse.Namespace) -> int:
    """Worst-case HTTP requests including API retries."""
    rollouts = args.max_episodes * (1 + args.continuations)
    return rollouts * args.max_turns * API_MAX_RETRIES


def load_episodes(path: Path, limit: int) -> list[EpisodeJSONL]:
    episodes: list[EpisodeJSONL] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            episodes.append(EpisodeJSONL.model_validate_json(line))
            if len(episodes) == limit:
                break
    if not episodes:
        raise ValueError("episodes file contains no records")
    return episodes


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


async def collect(args: argparse.Namespace) -> list[dict]:
    validate_args(args)
    episodes = load_episodes(args.episodes, args.max_episodes)
    policy = ContinuationPolicy(
        model=args.model,
        sampling_args={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
    )
    commit = current_commit()
    records: list[dict] = []

    total_rollouts = len(episodes) * (1 + args.continuations)
    request_limit = total_rollouts * args.max_turns * API_MAX_RETRIES
    print(
        f"Collecting {len(episodes)} prefix value(s): {total_rollouts} rollouts, "
        f"at most {request_limit} provider requests "
        f"({args.continuations} continuations each)."
    )

    for episode_index, episode in enumerate(episodes):
        csv_source = str(args.csv or Path(episode.csv_source))
        source_seed = args.seed + episode_index * (args.continuations + 1)
        source = await run_initial_model_trace(
            csv_source=csv_source,
            question_text=episode.question["question_text"],
            policy=policy,
            max_turns=args.max_turns,
            seed=source_seed,
        )
        if len(source.trace["turns"]) <= args.turn_count:
            raise ValueError(
                f"episode {episode.episode_id} produced only "
                f"{len(source.trace['turns'])} turn(s); cannot select boundary "
                f"after turn {args.turn_count}"
            )

        prefix = build_trajectory_prefix(
            episode_id=episode.episode_id,
            csv_source=csv_source,
            system_prompt=source.system_prompt,
            question_text=episode.question["question_text"],
            trace=source.trace,
            turn_responses=source.turn_responses,
            turn_completed=source.turn_completed,
            conversation_messages=(
                source.boundary_messages[args.turn_count - 1] if args.turn_count else []
            ),
            turn_count=args.turn_count,
            max_turns=args.max_turns,
        )
        continuation_seeds = [
            source_seed + offset for offset in range(1, args.continuations + 1)
        ]
        record = await collect_prefix_value(
            prefix=prefix,
            question=episode.question,
            policy=policy,
            seeds=continuation_seeds,
            float_tolerance=args.float_tolerance,
            code_commit=commit,
            dataset_revision=args.dataset_revision,
        )
        records.append(record.model_dump(mode="json"))
        print(
            f"{episode.episode_id}: value={record.value}, "
            f"labeled={record.labeled_continuations}/"
            f"{record.attempted_continuations}"
        )

    return records


async def main() -> None:
    args = parse_args()
    records = await collect(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Wrote {len(records)} record(s) to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

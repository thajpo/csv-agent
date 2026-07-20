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
MAX_CANDIDATES_PER_EPISODE = 8
MAX_SOURCE_ATTEMPTS_PER_CANDIDATE = 3
MAX_EPISODES = 64
MAX_TURNS = 20
MAX_PROVIDER_REQUESTS = 10_000

FIRST_ACTION_SUFFIX = """VALUE EXPERIMENT RULE:
On your first turn, do not call submit(). Execute exactly one useful Python
step and print what you learn. On later turns, finish the task and call
submit() when you have the answer."""


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
    parser.add_argument("--candidates-per-episode", type=int, default=3)
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
    if not 1 <= args.candidates_per_episode <= MAX_CANDIDATES_PER_EPISODE:
        raise ValueError(
            "candidates-per-episode must be between 1 and "
            f"{MAX_CANDIDATES_PER_EPISODE}"
        )
    request_limit = maximum_provider_requests(args)
    if request_limit > MAX_PROVIDER_REQUESTS:
        raise ValueError(
            f"configuration permits up to {request_limit} provider requests; "
            f"the canary limit is {MAX_PROVIDER_REQUESTS}"
        )


def maximum_provider_requests(args: argparse.Namespace) -> int:
    """Worst-case HTTP requests including API retries."""
    source_requests = (
        args.max_episodes
        * args.candidates_per_episode
        * MAX_SOURCE_ATTEMPTS_PER_CANDIDATE
        * max(args.turn_count, 1)
    )
    continuation_requests = (
        args.max_episodes
        * args.candidates_per_episode
        * args.continuations
        * args.max_turns
    )
    return (source_requests + continuation_requests) * API_MAX_RETRIES


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
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src",
            "packages",
            "scripts",
            "pyproject.toml",
            "uv.lock",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("refusing to collect with uncommitted code changes")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def select_boundary(source, turn_count: int, max_turns: int) -> tuple[list[dict], int]:
    if turn_count > len(source.trace["turns"]):
        raise ValueError(
            f"source produced only {len(source.trace['turns'])} executed turn(s); "
            f"cannot select boundary after executed turn {turn_count}"
        )
    if turn_count == 0:
        return [], 0

    boundary_index = turn_count - 1
    if source.turn_completed[boundary_index]:
        raise ValueError("selected boundary is terminal")
    consumed_turns = source.boundary_consumed_turns[boundary_index]
    if consumed_turns >= max_turns:
        raise ValueError("selected boundary has no remaining turn budget")
    return source.boundary_messages[boundary_index], consumed_turns


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

    total_prefixes = len(episodes) * args.candidates_per_episode
    total_rollouts = total_prefixes * (1 + args.continuations)
    request_limit = maximum_provider_requests(
        argparse.Namespace(**{**vars(args), "max_episodes": len(episodes)})
    )
    print(
        f"Collecting {total_prefixes} prefix value(s): {total_rollouts} rollouts, "
        f"at most {request_limit} provider requests "
        f"({args.continuations} continuations each)."
    )

    for episode_index, episode in enumerate(episodes):
        csv_source = str(args.csv or Path(episode.csv_source))
        accepted_prefix_ids: set[str] = set()
        source_attempt = 0
        while len(accepted_prefix_ids) < args.candidates_per_episode:
            if source_attempt >= (
                args.candidates_per_episode * MAX_SOURCE_ATTEMPTS_PER_CANDIDATE
            ):
                raise RuntimeError(
                    f"could not collect {args.candidates_per_episode} distinct "
                    f"nonterminal candidates for {episode.episode_id}"
                )
            source_seed = (
                args.seed
                + episode_index * 100_000
                + source_attempt * (args.continuations + 1)
            )
            source_attempt += 1
            source = await run_initial_model_trace(
                csv_source=csv_source,
                question_text=episode.question["question_text"],
                policy=policy,
                max_turns=max(args.turn_count, 1),
                seed=source_seed,
                system_prompt_suffix=FIRST_ACTION_SUFFIX,
            )
            try:
                boundary_messages, consumed_turns = select_boundary(
                    source, args.turn_count, args.max_turns
                )
            except ValueError as error:
                print(f"Skipping source attempt for {episode.episode_id}: {error}")
                continue

            prefix = build_trajectory_prefix(
                episode_id=episode.episode_id,
                csv_source=csv_source,
                system_prompt=source.system_prompt,
                question_text=episode.question["question_text"],
                trace=source.trace,
                turn_responses=source.turn_responses,
                turn_completed=source.turn_completed,
                conversation_messages=boundary_messages,
                turn_count=args.turn_count,
                consumed_turns=consumed_turns,
                max_turns=args.max_turns,
            )
            if prefix.prefix_id in accepted_prefix_ids:
                print(f"Skipping duplicate source attempt for {episode.episode_id}")
                continue
            accepted_prefix_ids.add(prefix.prefix_id)
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
                f"{episode.episode_id} candidate {len(accepted_prefix_ids)}: "
                f"value={record.value}, labeled={record.labeled_continuations}/"
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

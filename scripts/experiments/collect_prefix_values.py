"""Collect a small verifier-grounded value estimate for one turn boundary.

This command intentionally defaults to one episode and caps continuation
branches. It reports the planned rollout count before starting; each rollout
can require multiple turn-level model requests.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from csv_spec import (
    ContinuationPolicy,
    EpisodeJSONL,
    PrefixValueCollectionContract,
    PrefixValueRecord,
)

from src.value import (
    build_trajectory_prefix,
    collect_prefix_value,
    run_initial_model_trace,
    run_model_continuation,
)
from src.core.model import API_MAX_RETRIES

MAX_CONTINUATIONS = 16
MAX_CANDIDATES_PER_EPISODE = 8
MAX_SOURCE_ATTEMPTS_PER_CANDIDATE = 3
MAX_EPISODES = 64
MAX_TURNS = 20
MAX_PROVIDER_REQUESTS = 10_000
MAX_CONCURRENCY = 16

FIRST_ACTION_SUFFIX = """VALUE EXPERIMENT RULE:
On your first turn, do not call submit(). Execute exactly one useful Python
step that directly advances the question and print what you learn. The data
overview already shows columns, types, example rows, and basic summaries, so do
not spend the turn only listing column names. On later turns, finish the task
and call submit() when you have the answer."""


def candidate_request(candidate_index: int) -> str:
    strategies = (
        "Choose one useful prerequisite selection or data-quality check.",
        "Choose a necessary calculation or transformation, not another broad inspection.",
        "Choose a direct partial result or consistency check that moves closer to the final answer.",
    )
    strategy = strategies[min(candidate_index, len(strategies) - 1)]
    return (
        "Begin with one concrete, question-relevant intermediate action. Your "
        "response must contain 1-3 sentences explaining that action, followed by "
        "exactly one fenced ```python code block that prints what it learns. Do "
        "not call submit() anywhere, do not solve the whole task in this turn, and "
        f"do not only list the column names. {strategy} Do not repeat or merely "
        "reformat another candidate."
    )


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
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--float-tolerance", type=float, default=0.1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep valid records already present in --output and collect the rest.",
    )
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
    if args.turn_count == 0 and args.candidates_per_episode != 1:
        raise ValueError("initial-state collection requires one candidate per episode")
    if not 1 <= args.continuations <= MAX_CONTINUATIONS:
        raise ValueError(f"continuations must be between 1 and {MAX_CONTINUATIONS}")
    if not 1 <= args.candidates_per_episode <= MAX_CANDIDATES_PER_EPISODE:
        raise ValueError(
            f"candidates-per-episode must be between 1 and {MAX_CANDIDATES_PER_EPISODE}"
        )
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be between 1 and {MAX_CONCURRENCY}")
    request_limit = maximum_provider_requests(args)
    if request_limit > MAX_PROVIDER_REQUESTS:
        raise ValueError(
            f"configuration permits up to {request_limit} provider requests; "
            f"the canary limit is {MAX_PROVIDER_REQUESTS}"
        )


def maximum_provider_requests(args: argparse.Namespace) -> int:
    """Worst-case HTTP requests including API retries."""
    source_attempts = (
        args.max_episodes
        * args.candidates_per_episode
        * MAX_SOURCE_ATTEMPTS_PER_CANDIDATE
    )
    source_requests = source_attempts * max(args.turn_count, 1)
    continuation_requests = source_attempts * args.continuations * args.max_turns
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


def first_action_identity(source) -> str:
    return prefix_action_identity(source.trace["turns"])


class _BindingScope:
    def __init__(self, kind: str, parent: "_BindingScope | None" = None) -> None:
        self.kind = kind
        self.parent = parent
        self.bindings: dict[str, str] = {}
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()

    def bind(self, name: str) -> None:
        if name in self.global_names or name in self.nonlocal_names:
            return
        self.bindings.setdefault(name, f"local_{len(self.bindings)}")

    def root(self) -> "_BindingScope":
        scope = self
        while scope.parent is not None:
            scope = scope.parent
        return scope

    def resolve(self, name: str) -> str:
        scope: _BindingScope | None = self
        skip_class_scopes = self.kind in {"function", "lambda", "comprehension"}
        while scope is not None:
            if name in scope.global_names:
                return scope.root().bindings.get(name, name)
            if name in scope.bindings:
                return scope.bindings[name]
            scope = scope.parent
            if skip_class_scopes and scope is not None and scope.kind == "class":
                scope = scope.parent
        return name


class _BindingCollector(ast.NodeVisitor):
    def __init__(self, tree: ast.AST) -> None:
        self.root = _BindingScope("module")
        self.current = self.root
        self.scopes: dict[ast.AST, _BindingScope] = {tree: self.root}

    def _enter(self, node: ast.AST, kind: str) -> _BindingScope:
        scope = _BindingScope(kind, self.current)
        self.scopes[node] = scope
        self.current = scope
        return scope

    def _visit_function_context(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        self.current.bind(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.args.vararg and node.args.vararg.annotation is not None:
            self.visit(node.args.vararg.annotation)
        if node.args.kwarg and node.args.kwarg.annotation is not None:
            self.visit(node.args.kwarg.annotation)
        if node.returns is not None:
            self.visit(node.returns)
        parent = self.current
        self._enter(node, "function")
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            self.current.bind(argument.arg)
        if node.args.vararg:
            self.current.bind(node.args.vararg.arg)
        if node.args.kwarg:
            self.current.bind(node.args.kwarg.arg)
        for statement in node.body:
            self.visit(statement)
        self.current = parent

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_context(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_context(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        parent = self.current
        self._enter(node, "lambda")
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            self.current.bind(argument.arg)
        if node.args.vararg:
            self.current.bind(node.args.vararg.arg)
        if node.args.kwarg:
            self.current.bind(node.args.kwarg.arg)
        self.visit(node.body)
        self.current = parent

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.current.bind(node.name)
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        parent = self.current
        self._enter(node, "class")
        for statement in node.body:
            self.visit(statement)
        self.current = parent

    def _visit_comprehension_scope(
        self, node: ast.AST, result_nodes: tuple[ast.AST, ...]
    ) -> None:
        generators = node.generators
        self.visit(generators[0].iter)
        parent = self.current
        self._enter(node, "comprehension")
        self.visit(generators[0].target)
        for condition in generators[0].ifs:
            self.visit(condition)
        for generator in generators[1:]:
            self.visit(generator.iter)
            self.visit(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        for result in result_nodes:
            self.visit(result)
        self.current = parent

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension_scope(node, (node.elt,))

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension_scope(node, (node.elt,))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension_scope(node, (node.elt,))

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension_scope(node, (node.key, node.value))

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.current.bind(node.id)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        scope = self.current
        while scope.kind == "comprehension" and scope.parent is not None:
            scope = scope.parent
        scope.bind(node.target.id)

    def visit_Global(self, node: ast.Global) -> None:
        self.current.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.current.nonlocal_names.update(node.names)

    def visit_Import(self, node: ast.Import) -> None:
        for imported in node.names:
            self.current.bind(imported.asname or imported.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for imported in node.names:
            if imported.name != "*":
                self.current.bind(imported.asname or imported.name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self.current.bind(node.name)
        for statement in node.body:
            self.visit(statement)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.pattern is not None:
            self.visit(node.pattern)
        if node.name is not None:
            self.current.bind(node.name)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name is not None:
            self.current.bind(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        for pattern in node.patterns:
            self.visit(pattern)
        if node.rest is not None:
            self.current.bind(node.rest)


class _BindingNormalizer(ast.NodeTransformer):
    def __init__(self, collector: _BindingCollector) -> None:
        self.current = collector.root
        self.scopes = collector.scopes

    def _visit_arguments(self, arguments: ast.arguments) -> ast.arguments:
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            argument.arg = self.current.resolve(argument.arg)
        if arguments.vararg:
            arguments.vararg.arg = self.current.resolve(arguments.vararg.arg)
        if arguments.kwarg:
            arguments.kwarg.arg = self.current.resolve(arguments.kwarg.arg)
        return arguments

    def _visit_function_context(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        node.name = self.current.resolve(node.name)
        node.decorator_list = [self.visit(item) for item in node.decorator_list]
        node.args.defaults = [self.visit(item) for item in node.args.defaults]
        node.args.kw_defaults = [
            self.visit(item) if item is not None else None
            for item in node.args.kw_defaults
        ]
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            if argument.annotation is not None:
                argument.annotation = self.visit(argument.annotation)
        if node.args.vararg and node.args.vararg.annotation is not None:
            node.args.vararg.annotation = self.visit(node.args.vararg.annotation)
        if node.args.kwarg and node.args.kwarg.annotation is not None:
            node.args.kwarg.annotation = self.visit(node.args.kwarg.annotation)
        if node.returns is not None:
            node.returns = self.visit(node.returns)
        parent = self.current
        self.current = self.scopes[node]
        node.args = self._visit_arguments(node.args)
        node.body = [self.visit(item) for item in node.body]
        self.current = parent
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return self._visit_function_context(node)

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        return self._visit_function_context(node)

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        node.args.defaults = [self.visit(item) for item in node.args.defaults]
        node.args.kw_defaults = [
            self.visit(item) if item is not None else None
            for item in node.args.kw_defaults
        ]
        parent = self.current
        self.current = self.scopes[node]
        node.args = self._visit_arguments(node.args)
        node.body = self.visit(node.body)
        self.current = parent
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.name = self.current.resolve(node.name)
        node.decorator_list = [self.visit(item) for item in node.decorator_list]
        node.bases = [self.visit(item) for item in node.bases]
        node.keywords = [self.visit(item) for item in node.keywords]
        parent = self.current
        self.current = self.scopes[node]
        node.body = [self.visit(item) for item in node.body]
        self.current = parent
        return node

    def _visit_comprehension_scope(self, node: ast.AST, result_fields: tuple[str, ...]):
        generators = node.generators
        generators[0].iter = self.visit(generators[0].iter)
        parent = self.current
        self.current = self.scopes[node]
        generators[0].target = self.visit(generators[0].target)
        generators[0].ifs = [self.visit(item) for item in generators[0].ifs]
        for generator in generators[1:]:
            generator.iter = self.visit(generator.iter)
            generator.target = self.visit(generator.target)
            generator.ifs = [self.visit(item) for item in generator.ifs]
        for field in result_fields:
            setattr(node, field, self.visit(getattr(node, field)))
        self.current = parent
        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.ListComp:
        return self._visit_comprehension_scope(node, ("elt",))

    def visit_SetComp(self, node: ast.SetComp) -> ast.SetComp:
        return self._visit_comprehension_scope(node, ("elt",))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.GeneratorExp:
        return self._visit_comprehension_scope(node, ("elt",))

    def visit_DictComp(self, node: ast.DictComp) -> ast.DictComp:
        return self._visit_comprehension_scope(node, ("key", "value"))

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self.current.resolve(node.id)
        return node

    def visit_Global(self, node: ast.Global) -> ast.Global:
        root = self.current.root()
        node.names = [root.bindings.get(name, name) for name in node.names]
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.Nonlocal:
        node.names = [self.current.resolve(name) for name in node.names]
        return node

    def visit_Import(self, node: ast.Import) -> ast.Import:
        for imported in node.names:
            bound = imported.asname or imported.name.split(".")[0]
            imported.asname = self.current.resolve(bound)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        for imported in node.names:
            if imported.name != "*":
                imported.asname = self.current.resolve(imported.asname or imported.name)
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.ExceptHandler:
        if node.type is not None:
            node.type = self.visit(node.type)
        if node.name is not None:
            node.name = self.current.resolve(node.name)
        node.body = [self.visit(item) for item in node.body]
        return node

    def visit_MatchAs(self, node: ast.MatchAs) -> ast.MatchAs:
        if node.pattern is not None:
            node.pattern = self.visit(node.pattern)
        if node.name is not None:
            node.name = self.current.resolve(node.name)
        return node

    def visit_MatchStar(self, node: ast.MatchStar) -> ast.MatchStar:
        if node.name is not None:
            node.name = self.current.resolve(node.name)
        return node

    def visit_MatchMapping(self, node: ast.MatchMapping) -> ast.MatchMapping:
        node.patterns = [self.visit(item) for item in node.patterns]
        if node.rest is not None:
            node.rest = self.current.resolve(node.rest)
        return node


def action_identity(code: str) -> str:
    tree = ast.parse(code)
    collector = _BindingCollector(tree)
    collector.visit(tree)
    normalized = _BindingNormalizer(collector).visit(tree)
    return ast.dump(normalized, include_attributes=False)


def prefix_action_identity(turns: Sequence[dict]) -> str:
    if not turns:
        return "initial-state"
    return action_identity(turns[0]["code"])


def load_existing_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def csv_source_for_episode(args: argparse.Namespace, episode: EpisodeJSONL) -> str:
    return str(args.csv or Path(episode.csv_source))


def build_collection_contract(
    args: argparse.Namespace,
    episodes: Sequence[EpisodeJSONL],
    policy: ContinuationPolicy,
    code_commit: str,
) -> PrefixValueCollectionContract:
    episode_inputs = []
    for episode in episodes:
        csv_source = csv_source_for_episode(args, episode)
        csv_sha256 = hashlib.sha256(Path(csv_source).read_bytes()).hexdigest()
        episode_inputs.append(
            {
                "episode_id": episode.episode_id,
                "question": episode.question,
                "csv_source": csv_source,
                "csv_sha256": csv_sha256,
            }
        )
    serialized_inputs = json.dumps(
        episode_inputs,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return PrefixValueCollectionContract(
        code_commit=code_commit,
        dataset_revision=args.dataset_revision,
        policy=policy,
        episode_inputs_hash=hashlib.sha256(serialized_inputs.encode()).hexdigest(),
        turn_count=args.turn_count,
        max_turns=args.max_turns,
        continuations=args.continuations,
        candidates_per_episode=args.candidates_per_episode,
        seed=args.seed,
        float_tolerance=args.float_tolerance,
    )


def validate_resume_records(
    records: Sequence[dict],
    *,
    episode_ids: set[str],
    contract: PrefixValueCollectionContract,
) -> list[PrefixValueRecord]:
    actions_by_episode: dict[str, set[str]] = {}
    validated_records: list[PrefixValueRecord] = []
    for payload in records:
        try:
            record = PrefixValueRecord.model_validate(payload)
        except Exception as error:
            raise ValueError("resume record is invalid") from error
        episode_id = record.prefix.episode_id
        if episode_id not in episode_ids:
            raise ValueError(f"resume record has unexpected episode: {episode_id}")
        if record.collection_contract != contract:
            raise ValueError("resume record collection contract does not match")
        if not record.prefix.turns and contract.turn_count != 0:
            raise ValueError("resume record has no first action")
        if record.labeled_continuations == record.attempted_continuations:
            try:
                action_id = prefix_action_identity(record.prefix.turns)
            except (KeyError, SyntaxError, TypeError) as error:
                raise ValueError("resume record has an invalid first action") from error
            actions = actions_by_episode.setdefault(episode_id, set())
            if action_id in actions:
                raise ValueError(f"resume records repeat an action for {episode_id}")
            actions.add(action_id)
            if len(actions) > contract.candidates_per_episode:
                raise ValueError(f"too many resume records for {episode_id}")
        validated_records.append(record)
    return validated_records


async def collect(
    args: argparse.Namespace,
    *,
    record_sink: Callable[[dict], None] | None = None,
    existing_records: Sequence[dict] = (),
) -> list[dict]:
    validate_args(args)
    episodes = load_episodes(args.episodes, args.max_episodes)
    policy = ContinuationPolicy(
        model=args.model,
        sampling_args={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
        request_timeout_seconds=args.request_timeout_seconds,
    )
    commit = current_commit()
    contract = build_collection_contract(args, episodes, policy, commit)
    episode_ids = {episode.episode_id for episode in episodes}
    validated_existing = validate_resume_records(
        existing_records,
        episode_ids=episode_ids,
        contract=contract,
    )
    existing_by_episode = {episode_id: [] for episode_id in episode_ids}
    prior_attempts_by_episode = {episode_id: 0 for episode_id in episode_ids}
    for record in validated_existing:
        prior_attempts_by_episode[record.prefix.episode_id] += 1
        if record.labeled_continuations == record.attempted_continuations:
            existing_by_episode[record.prefix.episode_id].append(
                record.model_dump(mode="json")
            )
    rollout_slots = asyncio.Semaphore(args.concurrency)

    async def limited_continuation(prefix, continuation_policy, index, seed):
        async with rollout_slots:
            return await run_model_continuation(
                prefix, continuation_policy, index, seed
            )

    completed_existing = sum(len(records) for records in existing_by_episode.values())
    total_prefixes = len(episodes) * args.candidates_per_episode - completed_existing
    total_rollouts = total_prefixes * (1 + args.continuations)
    request_limit = maximum_provider_requests(
        argparse.Namespace(**{**vars(args), "max_episodes": len(episodes)})
    )
    print(
        f"Collecting {total_prefixes} prefix value(s): {total_rollouts} rollouts, "
        f"at most {request_limit} provider requests "
        f"({args.continuations} continuations each)."
    )

    async def collect_episode(episode_index, episode) -> list[dict]:
        episode_records = list(existing_by_episode[episode.episode_id])
        csv_source = csv_source_for_episode(args, episode)
        accepted_action_ids = {
            prefix_action_identity(record["prefix"]["turns"])
            for record in episode_records
        }
        source_attempt = 0
        resume_seed_offset = prior_attempts_by_episode[episode.episode_id] * 10_000_000
        while len(accepted_action_ids) < args.candidates_per_episode:
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
                + resume_seed_offset
                + source_attempt * (args.continuations + 1)
            )
            source_attempt += 1
            try:
                async with rollout_slots:
                    source = await run_initial_model_trace(
                        csv_source=csv_source,
                        question_text=episode.question["question_text"],
                        policy=policy,
                        max_turns=max(args.turn_count, 1),
                        seed=source_seed,
                        system_prompt_suffix=FIRST_ACTION_SUFFIX,
                        initial_user_message=candidate_request(
                            len(accepted_action_ids)
                        ),
                    )
            except Exception as error:
                print(
                    f"Skipping source attempt for {episode.episode_id}: "
                    f"{type(error).__name__}: {error}"
                )
                continue
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
            action_id = first_action_identity(source)
            if action_id in accepted_action_ids:
                print(f"Skipping duplicate source attempt for {episode.episode_id}")
                continue
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
                collection_contract=contract,
                runner=limited_continuation,
            )
            serialized = record.model_dump(mode="json")
            if record_sink is not None:
                record_sink(serialized)
            if record.labeled_continuations != record.attempted_continuations:
                print(
                    f"Retrying incomplete candidate for {episode.episode_id}: "
                    f"labeled={record.labeled_continuations}/"
                    f"{record.attempted_continuations}"
                )
                continue
            accepted_action_ids.add(action_id)
            episode_records.append(serialized)
            print(
                f"{episode.episode_id} candidate {len(accepted_action_ids)}: "
                f"value={record.value}, labeled={record.labeled_continuations}/"
                f"{record.attempted_continuations}"
            )
        return episode_records

    episode_results = await asyncio.gather(
        *(
            collect_episode(episode_index, episode)
            for episode_index, episode in enumerate(episodes)
        )
    )
    return [record for episode_records in episode_results for record in episode_records]


async def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing_records: list[dict] = []
    if args.resume:
        if not args.output.is_file():
            raise ValueError("--resume requires an existing --output file")
        existing_records = load_existing_records(args.output)
        print(f"Resuming from {len(existing_records)} existing record(s).")
    else:
        args.output.write_text("")

    def persist(record: dict) -> None:
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    records = await collect(
        args,
        record_sink=persist,
        existing_records=existing_records,
    )
    compacted = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    temporary_output = args.output.with_name(f".{args.output.name}.tmp")
    temporary_output.write_text(compacted, encoding="utf-8")
    temporary_output.replace(args.output)
    print(f"Wrote {len(records)} record(s) to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

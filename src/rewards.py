"""
Reward calculation for scavenger hunt training.

This module implements the state-matching reward system where students
are rewarded for matching intermediate data states (artifacts) created
by the teacher, not just for matching the final answer.
"""

from src.types import Artifact


def calculate_scavenger_hunt_reward(
    teacher_artifacts: dict[str, Artifact],
    student_artifacts: dict[str, Artifact],
    teacher_final_hash: str | None,
    student_final_hash: str | None
) -> tuple[list[str], bool, int, int]:
    """
    Calculate scavenger hunt rewards based on artifact matching.

    The "scavenger hunt" compares the hashes of DataFrames and scalars
    created by teacher vs student. Students get:
    - Dense reward (+1) for each intermediate artifact match
    - Sparse reward (+5) for final answer match

    Args:
        teacher_artifacts: Dict of {var_name: Artifact} from teacher trace
        student_artifacts: Dict of {var_name: Artifact} from student trace
        teacher_final_hash: Hash of teacher's submit() value
        student_final_hash: Hash of student's submit() value

    Returns:
        Tuple of:
        - intermediate_matches: List of matched artifact descriptions (e.g., "df_student ←→ df_teacher")
        - final_match: Whether final answers match
        - dense_reward: Number of intermediate matches
        - sparse_reward: 5 if final match, else 0

    Example:
        >>> teacher_arts = {'df_filtered': Artifact(name='df_filtered', hash='abc123', type='DataFrame')}
        >>> student_arts = {'df_filt': Artifact(name='df_filt', hash='abc123', type='DataFrame')}
        >>> matches, final, dense, sparse = calculate_scavenger_hunt_reward(
        ...     teacher_arts, student_arts, 'xyz789', 'xyz789'
        ... )
        >>> matches
        ['df_filt ←→ df_filtered']
        >>> dense, sparse
        (1, 5)
    """
    # Build hash → name mappings
    teacher_hash_to_name = {art.hash: art.name for art in teacher_artifacts.values()}
    student_hash_to_name = {art.hash: art.name for art in student_artifacts.values()}

    # Find intermediate artifact matches
    matched_names = []
    for student_hash, student_name in student_hash_to_name.items():
        if student_hash in teacher_hash_to_name:
            teacher_name = teacher_hash_to_name[student_hash]
            matched_names.append(f"{student_name} ←→ {teacher_name}")

    dense_reward = len(matched_names)

    # Check final answer match
    final_match = False
    if teacher_final_hash and student_final_hash:
        final_match = (teacher_final_hash == student_final_hash)

    sparse_reward = 5 if final_match else 0

    return matched_names, final_match, dense_reward, sparse_reward


def calculate_total_reward(
    intermediate_matches: list[str],
    final_match: bool,
    dense_reward: int,
    sparse_reward: int,
    difficulty_multiplier: float = 1.0
) -> float:
    """
    Calculate total reward with optional difficulty scaling.

    Args:
        intermediate_matches: List of matched artifacts
        final_match: Whether final answer matched
        dense_reward: Number of intermediate matches
        sparse_reward: Reward for final match (typically 5)
        difficulty_multiplier: Scale reward by difficulty (e.g., 1.0 for MEDIUM, 1.5 for HARD, 2.0 for VERY_HARD)

    Returns:
        Total reward (float)

    Example:
        >>> total = calculate_total_reward(
        ...     intermediate_matches=['df_filt ←→ df_filtered'],
        ...     final_match=True,
        ...     dense_reward=1,
        ...     sparse_reward=5,
        ...     difficulty_multiplier=1.5
        ... )
        >>> total
        9.0
    """
    base_reward = dense_reward + sparse_reward
    return base_reward * difficulty_multiplier


def format_reward_summary(
    intermediate_matches: list[str],
    final_match: bool,
    dense_reward: int,
    sparse_reward: int,
    total_reward: float
) -> str:
    """
    Format reward information as a human-readable summary.

    Args:
        intermediate_matches: List of matched artifacts
        final_match: Whether final answer matched
        dense_reward: Number of intermediate matches
        sparse_reward: Reward for final match
        total_reward: Total combined reward

    Returns:
        Formatted summary string

    Example:
        >>> summary = format_reward_summary(
        ...     intermediate_matches=['df_filt ←→ df_filtered', 'mean_val ←→ mean_yield'],
        ...     final_match=True,
        ...     dense_reward=2,
        ...     sparse_reward=5,
        ...     total_reward=7.0
        ... )
        >>> print(summary)
        Reward Summary:
        ✓ Dense Reward: +2 (2 intermediate matches)
          - df_filt ←→ df_filtered
          - mean_val ←→ mean_yield
        ✓ Sparse Reward: +5 (final answer matched)
        Total: 7.0
    """
    lines = ["Reward Summary:"]

    # Dense reward
    if intermediate_matches:
        lines.append(f"✓ Dense Reward: +{dense_reward} ({len(intermediate_matches)} intermediate matches)")
        for match in intermediate_matches:
            lines.append(f"  - {match}")
    else:
        lines.append(f"✗ Dense Reward: 0 (no intermediate matches)")

    # Sparse reward
    if final_match:
        lines.append(f"✓ Sparse Reward: +{sparse_reward} (final answer matched)")
    else:
        lines.append(f"✗ Sparse Reward: 0 (final answer did not match)")

    # Total
    lines.append(f"Total: {total_reward}")

    return "\n".join(lines)

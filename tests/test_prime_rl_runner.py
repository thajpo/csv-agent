import json

from scripts.prime_rl import plot_run


def test_plot_run_collects_rollout_rewards_and_writes_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    step_dir = run_dir / "rollouts" / "step_000001"
    step_dir.mkdir(parents=True)
    (step_dir / "train_rollouts.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"reward": 0.25}),
                json.dumps({"reward": 0.75}),
            ]
        )
        + "\n"
    )
    (step_dir / "eval_rollouts_csv-agent-val.jsonl").write_text(
        json.dumps({"raw": {"reward": 1.0}}) + "\n"
    )

    points = plot_run.collect_points(run_dir)

    assert [(p.step, p.split, p.env, p.mean_reward, p.n) for p in points] == [
        (1, "eval", "csv-agent-val", 1.0, 1),
        (1, "train", "all", 0.5, 2),
    ]


def test_plot_run_main_writes_summary_files(tmp_path):
    run_dir = tmp_path / "run"
    step_dir = run_dir / "rollouts" / "step_000002"
    step_dir.mkdir(parents=True)
    (step_dir / "train_rollouts.jsonl").write_text(json.dumps({"reward": 0.5}) + "\n")
    artifact_dir = tmp_path / "summary"

    rc = plot_run.main(
        ["--run-dir", str(run_dir), "--artifact-dir", str(artifact_dir), "--title", "Test Run"]
    )

    assert rc == 0
    assert json.loads((artifact_dir / "metrics.json").read_text())[0]["mean_reward"] == 0.5
    assert "mean_reward" in (artifact_dir / "metrics.csv").read_text()
    assert "Test Run" in (artifact_dir / "reward_curve.svg").read_text()
    assert "Reward curve" in (artifact_dir / "README.md").read_text()

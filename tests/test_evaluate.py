import json

import numpy as np

from expert_data.checkpoint import save_checkpoint
from expert_data.config import Config, EnvConfig, PPOConfig, RunConfig
from expert_data.evaluate import main as evaluate_main
from expert_data.evaluate import wilson_interval
from expert_data.tracking import label_consistency, masked_label_consistency
from expert_data.train import build


def test_wilson_interval():
    lo, hi = wilson_interval(512, 512)
    assert abs(lo - 0.9926) < 1e-3 and hi == 1.0
    lo, hi = wilson_interval(50, 100)
    assert abs(lo - 0.404) < 1e-3 and abs(hi - 0.596) < 1e-3


def test_masked_consistency_matches_unmasked_and_finds_lag():
    rng = np.random.default_rng(0)
    c = np.cumsum(rng.standard_normal((200, 6, 4)), axis=0)
    r = np.roll(c, 3, axis=0)  # realised trails the command by 3 steps
    mask = np.ones((200, 6), bool)
    mask[:5] = False  # the roll wraps around at the start
    _, lag, r_best = masked_label_consistency(c, r, mask)
    assert np.all(lag == 3) and np.all(r_best > 0.999)
    # Same numbers as the unmasked version on full series (single world, pooled == per series).
    r0_a, lag_a, _ = label_consistency(c[:, :1], c[:, :1] * 2.0)
    r0_b, lag_b, _ = masked_label_consistency(c[:, :1], c[:, :1] * 2.0, np.ones((200, 1), bool))
    np.testing.assert_allclose(r0_a, r0_b, atol=1e-6)
    np.testing.assert_array_equal(lag_a, lag_b)


def test_evaluate_runs_on_an_untrained_checkpoint(tmp_path, monkeypatch):
    cfg = Config(
        env=EnvConfig(n_envs=4, n_gates=1, n_gates_min=1, track_type="chain"),
        ppo=PPOConfig(n_steps=4, n_epochs=1, n_minibatches=2, total_env_steps=16),
        run=RunConfig(device="cpu", jax_cache=False),
    )
    _, _, train_state, *_ = build(cfg)
    ckpt = tmp_path / "ckpt.pkl"
    save_checkpoint(
        ckpt,
        train_state,
        {"env": cfg.env.__dict__, "net": cfg.net.__dict__, "ppo": cfg.ppo.__dict__},
    )
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "eval.json"
    summary = evaluate_main(
        [
            "--ckpt",
            str(ckpt),
            "--n-episodes",
            "4",
            "--device",
            "cpu",
            "--target-speed",
            "1.2",
            "--json",
            str(out),
            "--save-traj",
            str(tmp_path / "traj.npz"),
            "--traj-worlds",
            "2",
        ]
    )
    assert summary["checkpoint"] == "ckpt.pkl"  # relative, never an absolute path
    assert summary["policy"] == "tanh"
    np.testing.assert_allclose(summary["target_speed"], [1.2, 1.2], rtol=1e-6)
    rates = [summary[k] for k in ("success_rate", "crash_rate", "miss_rate", "timeout_rate")]
    assert abs(sum(rates) - 1.0) < 1e-6
    assert set(summary["saturation_ge_0.99"]) == {"vx", "vy", "vz", "wz"}
    assert json.loads(out.read_text())["n_episodes"] == 4
    assert "/" not in out.read_text().split('"checkpoint": ')[1].split(",")[0].strip('"')
    traj = np.load(tmp_path / "traj.npz")
    assert traj["pos"].shape[1] == 2 and traj["cmd"].shape[-1] == 4


def test_console_script_does_not_return_the_summary():
    # A console script passes main()'s return value to sys.exit: a dict would mean exit code 1.
    import tomllib
    from pathlib import Path

    from expert_data import evaluate

    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert pyproject["project"]["scripts"]["expert-eval"] == "expert_data.evaluate:cli"
    assert evaluate.cli.__annotations__.get("return") in (None, "None")

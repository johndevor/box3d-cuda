"""Local (cpu + serial lane) tests for the single-process GPU trainer.

Runs walk.train.gpu_train end to end against the SERIAL dwc1 dylib built by
walk.env.cuda_lane.build_library() — the same wrapper the CUDA .so uses on
the GPU host — so everything except the device is exercised locally.

If the working tree is mid-ABI-migration (duck_cuda sources bumped past the
committed wrapper's pinned version), the serial library is built from the
committed (HEAD) sources instead, keeping these tests runnable throughout.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from walk.env import cuda_lane
from walk.train import gpu_train, ppo

ENVS = 8
HORIZON = 16
SEED = 123

# Feed-forward 2-update metrics captured by running the PRE-gru trainer
# (walk/train/{gpu_train,ppo}.py at commit bbeb9174) with the exact argv()
# settings against the serial lane at the source digest below.
# test_ff_path_matches_pre_change_baseline asserts bitwise equality; it is
# only meaningful (and only runs) while the serial physics is unchanged.
FF_BASELINE = {
    "source_digest": "e0ab6fa01e1f62f4",
    "lines": [
        {
            "update": 1, "env_steps": 128,
            "reward_mean": 0.25269824266433716, "reward_std": 0.8367269039154053,
            "ep_return_mean": None, "ep_len_mean": None, "episodes": 0,
            "pi_loss": -0.07960005197674036, "v_loss": 3.320355851203203,
            "entropy": 12.859792470932007, "approx_kl": 0.03006874758284539,
            "clip_frac": 0.3828125,
        },
        {
            "update": 2, "env_steps": 256,
            "reward_mean": -0.04010656476020813, "reward_std": 0.9611939787864685,
            "ep_return_mean": -3.49450346454978, "ep_len_mean": 30.0, "episodes": 1,
            "pi_loss": -0.07139695761725307, "v_loss": 7.019042037427425,
            "entropy": 12.85073584318161, "approx_kl": 0.02270544320344925,
            "clip_frac": 0.271484375,
        },
    ],
}
BASELINE_KEYS = [k for k in FF_BASELINE["lines"][0]]

_LIB_CACHE: tuple[str, str] | None = None


def _head_digest(root: Path, rels: list[str]) -> str:
    h = hashlib.sha256()
    for rel in rels:
        h.update(rel.encode())
        h.update((root / rel).read_bytes())
    return h.hexdigest()[:16]


def serial_library() -> tuple[str, str]:
    """Return (library_path, source_digest) of a serial lane the current
    committed wrapper can load. Prefers the working tree; falls back to a
    build from HEAD sources when the tree is mid-ABI-migration."""
    global _LIB_CACHE
    if _LIB_CACHE is not None:
        return _LIB_CACHE
    try:
        path = cuda_lane.build_library()
        probe = cuda_lane.CudaDuckLane(1, library_path=path)
        probe.close()
        _LIB_CACHE = (str(path), cuda_lane._source_digest())
        return _LIB_CACHE
    except RuntimeError:
        pass  # ABI mismatch: another workstream bumped the sources
    root = cuda_lane.ROOT
    rels = [str(p.relative_to(root)) for p in cuda_lane._SOURCES]
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            f"git -C '{root}' archive HEAD experimental/duck_cuda | tar -x -C '{td}'",
            shell=True, check=True)
        tdp = Path(td)
        digest = _head_digest(tdp, rels)
        out = cuda_lane.BUILD_DIR / f"libduck_cuda_serial-HEAD-{digest}.dylib"
        if not out.is_file():
            cuda_lane.BUILD_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["clang++", *cuda_lane._FLAGS,
                 "-I", str(tdp / "experimental" / "duck_cuda" / "include"),
                 "-fPIC", "-shared",
                 str(tdp / "experimental" / "duck_cuda" / "src" / "duck_cuda_serial.cpp"),
                 "-o", str(out)],
                check=True, capture_output=True)
    probe = cuda_lane.CudaDuckLane(1, library_path=out)
    probe.close()
    _LIB_CACHE = (str(out), digest)
    return _LIB_CACHE


def argv(out: str, lib: str, **over) -> list[str]:
    base = {
        "envs": ENVS, "horizon": HORIZON, "updates": 3, "seed": SEED,
        "device": "cpu", "library": lib, "out": out,
        "preflight-steps": 30, "checkpoint-every": 2,
    }
    base.update(over)
    args = []
    for k, v in base.items():
        if v is None:
            continue
        args += [f"--{k}", str(v)]
    return args + ["--quiet"]


def read_metrics(out: Path) -> list[dict]:
    path = out / "metrics.jsonl"
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


class TestGpuTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib, cls.digest = serial_library()

    def test_three_updates_cpu_serial(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            metrics = gpu_train.main(argv(td, self.lib))
            train_lines = [m for m in metrics if "skipped" not in m]
            self.assertEqual(len(train_lines), 3)
            # metrics.jsonl written and parseable, rewards finite
            lines = read_metrics(out)
            self.assertEqual(len([m for m in lines if m["kind"] == "train"]), 3)
            for m in train_lines:
                self.assertTrue(math.isfinite(m["reward_mean"]),
                                f"non-finite reward at update {m['update']}: {m}")
                self.assertTrue(math.isfinite(m["pi_loss"]))
                self.assertTrue(math.isfinite(m["v_loss"]))
                self.assertGreater(m["steps_per_s"], 0.0)
                self.assertEqual(m["faults"], 0)
            self.assertEqual(train_lines[-1]["env_steps"], 3 * ENVS * HORIZON)
            # checkpoints: every 2 updates + latest at exit
            self.assertTrue((out / "ckpt_000002.pt").is_file())
            self.assertTrue((out / "latest.pt").is_file())
            ck = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(ck["update"], 3)
            # actor_final.pt: self-describing {"arch", "state_dict"}, cpu tensors
            obj = torch.load(out / "actor_final.pt", map_location="cpu",
                             weights_only=True)
            arch, sd = ppo.unpack_actor_file(obj)
            self.assertEqual(arch, "ff")
            actor = ppo.Actor(58, 14)
            actor.load_state_dict(sd)  # raises on any mismatch
            for v in sd.values():
                self.assertEqual(v.device.type, "cpu")

    def test_resume_runs_two_more_updates(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            gpu_train.main(argv(td, self.lib, updates=3))
            metrics = gpu_train.main(
                argv(td, self.lib, updates=5, resume=str(out / "latest.pt")))
            self.assertEqual([m["update"] for m in metrics], [4, 5])
            lines = [m for m in read_metrics(out) if m["kind"] == "train"]
            self.assertEqual([m["update"] for m in lines], [1, 2, 3, 4, 5])
            # env_steps carried across the resume
            self.assertEqual(lines[-1]["env_steps"], 5 * ENVS * HORIZON)
            ck = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(ck["update"], 5)

    def test_max_wall_s_exits_cleanly_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            t0 = time.perf_counter()
            metrics = gpu_train.main(
                argv(td, self.lib, updates=100000, **{"max-wall-s": 1}))
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 60.0, "wall-clock stop did not engage")
            self.assertLess(len(metrics), 100000)
            self.assertTrue((out / "latest.pt").is_file())
            self.assertTrue((out / "actor_final.pt").is_file())
            ck = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(ck["update"], len([m for m in metrics
                                                if "skipped" not in m]))


class TestGruPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib, cls.digest = serial_library()

    def test_gru_three_updates(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            metrics = gpu_train.main(argv(td, self.lib, policy="gru"))
            train_lines = [m for m in metrics if "skipped" not in m]
            self.assertEqual(len(train_lines), 3)
            for m in train_lines:
                for k in ("reward_mean", "pi_loss", "v_loss", "entropy", "approx_kl"):
                    self.assertTrue(math.isfinite(m[k]),
                                    f"non-finite {k} at update {m['update']}: {m}")
            # actor_final round-trip: arch recorded, weights identical to the
            # final checkpoint, loads into a fresh RecurrentActor and runs
            obj = torch.load(out / "actor_final.pt", map_location="cpu",
                             weights_only=True)
            arch, sd = ppo.unpack_actor_file(obj)
            self.assertEqual(arch, "gru")
            ck = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(ck["config"]["policy"], "gru")
            for k, v in ck["actor"].items():
                self.assertTrue(torch.equal(v, sd[k]), f"actor_final differs at {k}")
            actor = ppo.RecurrentActor(58, 14)
            actor.load_state_dict(sd)
            obs = torch.zeros(2, 58)
            act, h1 = actor.deterministic(obs, actor.initial_state(2))
            self.assertEqual(tuple(act.shape), (2, 14))
            self.assertEqual(tuple(h1.shape), (2, actor.gru_hidden))
            self.assertTrue(torch.isfinite(act).all())

    def test_hidden_state_resets_on_done(self):
        """An env that resets mid-window must see h == 0 on its next step."""
        E, H, done_step, done_env = 4, 6, 2, 1

        class ScriptedEnv:
            OBS, ACT = 58, 14

            def __init__(self):
                self.E = E
                self.t = 0

            def _obs(self):
                return np.full((E, self.OBS), 0.01 * (self.t + 1), np.float32)

            def reset(self, mask=None, seed=None):
                return self._obs()

            def step(self, action):
                self.t += 1
                done = np.zeros(E, bool)
                if self.t == done_step + 1:   # done AFTER loop index done_step
                    done[done_env] = True
                return self._obs(), np.ones(E, np.float32), done, {}

        class ProbeActor(ppo.RecurrentActor):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.h_seen = []

            def dist(self, obs, h):
                self.h_seen.append(h.detach().clone())
                return super().dist(obs, h)

        class ProbeCritic(ppo.RecurrentCritic):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.h_seen = []

            def forward(self, obs, h):
                self.h_seen.append(h.detach().clone())
                return super().forward(obs, h)

        torch.manual_seed(0)
        actor, critic = ProbeActor(58, 14), ProbeCritic(58)
        gen = torch.Generator()
        gen.manual_seed(0)
        env = ScriptedEnv()
        ep_ret, ep_len = np.zeros(E), np.zeros(E, np.int64)
        ro = gpu_train.collect_rollout_recurrent(
            env, actor, critic, env.reset(), actor.initial_state(E),
            critic.initial_state(E), H, gen, torch.device("cpu"), ep_ret, ep_len)
        for probe in (actor, critic):
            self.assertEqual(len(probe.h_seen), H)
            after = probe.h_seen[done_step + 1]      # h fed to the next step
            self.assertTrue((after[done_env] == 0).all(),
                            "hidden row of the reset env was not zeroed")
            others = [e for e in range(E) if e != done_env]
            for e in others:
                self.assertGreater(float(after[e].abs().sum()), 0.0,
                                   "live env hidden was unexpectedly zeroed")
            # and the reset env had NONZERO h just before its reset
            self.assertGreater(float(probe.h_seen[done_step][done_env].abs().sum()), 0.0)
        # stored window-initial hidden is all zeros here (fresh start)
        self.assertTrue((ro.h0_actor == 0).all() and (ro.h0_critic == 0).all())
        # episode bookkeeping saw exactly the one scripted reset
        self.assertEqual(len(ro.episodes), 1)
        self.assertEqual(ro.episodes[0][1], done_step + 1)

    def test_ff_path_matches_pre_change_baseline(self):
        if self.digest != FF_BASELINE["source_digest"]:
            self.skipTest(
                f"serial lane sources changed (digest {self.digest} != baseline "
                f"{FF_BASELINE['source_digest']}); ff bitwise baseline not comparable")
        with tempfile.TemporaryDirectory() as td:
            metrics = gpu_train.main(argv(td, self.lib, updates=2))
        self.assertEqual(len(metrics), 2)
        for got, want in zip(metrics, FF_BASELINE["lines"]):
            for k in BASELINE_KEYS:
                self.assertEqual(got[k], want[k],
                                 f"ff drift at update {want['update']} field {k}: "
                                 f"{got[k]!r} != {want[k]!r}")


class TestAcceptProbe(unittest.TestCase):
    """--accept-every N: strict-acceptance probe, auto-stop, accounting."""

    @classmethod
    def setUpClass(cls):
        cls.lib, cls.digest = serial_library()

    @staticmethod
    def _fake_trace(seconds: float) -> dict:
        n = int(round(seconds / 0.002))
        return {"schema": "duckgridwalk.episode/1", "dt": 0.002,
                "command_mps": 0.1, "seed": 4242, "resets": 0,
                "terminated": False, "truncated_at_horizon": True,
                "solver_fault": False,
                "ticks": {"time_s": [0.002 * (i + 1) for i in range(n)]}}

    class _StubProbeEnv:
        def close(self):
            pass

    def test_pass_confirm_stop_flow_and_accounting(self):
        calls, evals = [], []

        def fake_capture(env, policy, command=None, seconds=8.0, seed=None, **kw):
            calls.append((round(float(command), 2), float(seconds), seed))
            policy(np.zeros((1, 58), np.float32))    # exercise the ff wrapper
            return [self._fake_trace(seconds)]

        def fake_eval(trace):
            # confirmation episodes must arrive as the EXACT 8 s prefix
            evals.append(len(trace["ticks"]["time_s"]))
            return {"passed": True, "criteria": {}, "footfalls": []}

        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(gpu_train, "capture_episodes", fake_capture), \
                mock.patch.object(gpu_train, "evaluate_episode", fake_eval), \
                mock.patch.object(gpu_train, "_make_probe_env",
                                  lambda cfg, seed: self._StubProbeEnv()):
            out = Path(td)
            metrics = gpu_train.main(
                argv(td, self.lib, updates=10, **{"accept-every": 1}))
            train_lines = [m for m in metrics if m["kind"] == "train"]
            accept_lines = [m for m in metrics if m["kind"] == "accept"]
            # confirmed at update 1 -> training stopped immediately
            self.assertEqual(len(train_lines), 1)
            self.assertEqual(len(accept_lines), 1)
            self.assertTrue(accept_lines[0]["accepted"])
            self.assertTrue(accept_lines[0]["stage1_passed"])
            # stage-1 episodes (PROBE_SEEDS x 3 commands, 8 s) then 3
            # confirmation episodes (seed 4242, 11 s), in order
            n1 = 3 * len(gpu_train.PROBE_SEEDS)
            self.assertEqual(calls[:n1], [(c, 8.0, s) for s in gpu_train.PROBE_SEEDS
                                          for c in (0.10, 0.15, 0.20)])
            self.assertEqual(calls[n1:], [(c, 11.0, 4242)
                                          for c in (0.10, 0.15, 0.20)])
            self.assertEqual(evals, [4000] * (n1 + 3))  # all judged on 8 s ticks
            # acceptance artifacts + accounting fields
            acc = json.loads((out / "accepted" / "acceptance.json").read_text())
            self.assertTrue(acc["accepted"])
            self.assertEqual(acc["update"], 1)
            self.assertEqual(acc["env_steps"], ENVS * HORIZON)
            self.assertEqual(acc["policy"], "ff")
            self.assertGreaterEqual(acc["train_wall_s"], 0.0)
            self.assertGreater(acc["probe_wall_s"], 0.0)
            self.assertEqual(len(acc["episodes"]), n1)
            self.assertEqual(len(acc["confirmation"]), 3)
            arch, sd = ppo.unpack_actor_file(torch.load(
                out / "accepted" / "actor_accepted.pt", map_location="cpu",
                weights_only=True))
            self.assertEqual(arch, "ff")
            ppo.Actor(58, 14).load_state_dict(sd)
            ck = torch.load(out / "latest.pt", map_location="cpu",
                            weights_only=False)
            self.assertEqual(ck["update"], 1)
            self.assertIn("train_wall_s", ck)
            self.assertIn("probe_wall_s", ck)
            self.assertGreater(ck["probe_wall_s"], 0.0)

    def test_failing_probes_never_stop_training(self):
        n_probes = []

        def fake_capture(env, policy, command=None, seconds=8.0, seed=None, **kw):
            return [self._fake_trace(seconds)]

        def fake_eval(trace):
            n_probes.append(1)
            return {"passed": False,
                    "criteria": {"at_least_6_qualified_footfalls":
                                 {"pass": False, "detail": 0}},
                    "footfalls": []}

        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(gpu_train, "capture_episodes", fake_capture), \
                mock.patch.object(gpu_train, "evaluate_episode", fake_eval), \
                mock.patch.object(gpu_train, "_make_probe_env",
                                  lambda cfg, seed: self._StubProbeEnv()):
            out = Path(td)
            metrics = gpu_train.main(
                argv(td, self.lib, updates=3, **{"accept-every": 1}))
            train_lines = [m for m in metrics if m["kind"] == "train"]
            accept_lines = [m for m in metrics if m["kind"] == "accept"]
            self.assertEqual(len(train_lines), 3)    # ran to completion
            self.assertEqual(len(accept_lines), 3)   # probed every update
            self.assertTrue(all(not m["accepted"] for m in accept_lines))
            self.assertFalse((out / "accepted").exists())
            # short-circuit cost control: 1 failing episode per probe
            self.assertEqual(len(n_probes), 3)
            ck = torch.load(out / "latest.pt", map_location="cpu",
                            weights_only=False)
            self.assertEqual(ck["update"], 3)

    def test_prefix_truncation_roundtrips_real_trace(self):
        """Real serial-lane 11 s capture -> 8 s prefix passes the evaluator's
        integrity gate (structurally valid single 8 s episode)."""
        from walk.env import flat as flat_mod
        from walk.env.flat import FlatFloorDuckEnv
        from walk.eval.gait import evaluate_episode as real_eval

        old_horizon = flat_mod.HORIZON_STEPS
        env = FlatFloorDuckEnv(
            environments=1, seed=4242, perturbation_rad=0.0,
            lane_factory=lambda E, off: cuda_lane.CudaDuckLane(
                E, joint_offsets=off, library_path=self.lib))
        try:
            def policy(obs):                         # hold HOME: duck stands
                return np.zeros((1, 14), np.float32)
            with gpu_train._extended_horizon(int(round(11.0 / 0.02))):
                trace = gpu_train.capture_episodes(
                    env, policy, command=0.10, seconds=11.0, seed=4242)[0]
        finally:
            env.close()
        self.assertEqual(flat_mod.HORIZON_STEPS, old_horizon)  # restored
        # survived the full 11 s
        self.assertFalse(trace["terminated"])
        self.assertFalse(trace["solver_fault"])
        self.assertTrue(trace["truncated_at_horizon"])
        self.assertEqual(len(trace["ticks"]["time_s"]), 5500)
        # exact-8 s prefix round-trips the strict evaluator's integrity gate
        pre = gpu_train.truncate_trace_to_8s(trace)
        self.assertEqual(len(pre["ticks"]["time_s"]), 4000)
        for arr in pre["ticks"].values():
            self.assertEqual(len(arr), 4000)
        r = real_eval(pre)
        self.assertFalse(r["rejected"], r["criteria"])
        self.assertTrue(
            r["criteria"]["single_episode_no_reset_or_failure"]["pass"])
        # a standing duck is structurally valid but must not pass the gait
        self.assertFalse(r["passed"])


if __name__ == "__main__":
    unittest.main()

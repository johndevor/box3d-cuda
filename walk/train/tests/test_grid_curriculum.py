"""Tests for the cube-terrain curriculum runner (walk/train/grid_curriculum.py).

Run: .venv/bin/python -B -m unittest walk.train.tests.test_grid_curriculum
"""
import json
import tempfile
import unittest
from pathlib import Path

from walk.train import grid_curriculum as gc


def make_args(td, **overrides):
    argv = ["--out", str(Path(td) / "cur")]
    for k, v in overrides.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return gc.build_argparser().parse_args(argv)


class TestWarmstart(unittest.TestCase):
    def _tiny_actor(self, td):
        import torch
        from walk.train.ppo import Actor
        path = Path(td) / "actor.pt"
        actor = Actor(58, 14)
        torch.save({"arch": "ff", "state_dict": actor.state_dict()}, path)
        return path, actor

    def test_checkpoint_is_run_py_resumable(self):
        import torch
        from walk.train.ppo import PPOConfig, make_nets
        from walk.train.vec import derive_seed
        with tempfile.TemporaryDirectory() as td:
            src, actor = self._tiny_actor(td)
            out = gc.make_warmstart_checkpoint(src, Path(td) / "warm.pt",
                                               seed=917, lr=7.5e-5)
            ck = torch.load(out, map_location="cpu", weights_only=False)
            # exactly the fields run.py's resume path reads
            for key in ("update", "actor", "critic", "optimizer", "gen_state",
                        "env_steps", "faults_total"):
                self.assertIn(key, ck)
            self.assertEqual(ck["update"], 0)
            self.assertEqual(ck["env_steps"], 0)
            # actor weights survive the round trip bit-exactly
            for k, v in actor.state_dict().items():
                self.assertTrue(torch.equal(ck["actor"][k], v), k)
            # run.py's resume sequence works on fresh nets
            a2, c2 = make_nets(58, 14, PPOConfig())
            a2.load_state_dict(ck["actor"])
            c2.load_state_dict(ck["critic"])
            opt = torch.optim.Adam(list(a2.parameters()) + list(c2.parameters()),
                                   lr=7.5e-5)
            opt.load_state_dict(ck["optimizer"])
            gen = torch.Generator()
            gen.set_state(ck["gen_state"])
            # generator seeded exactly like a fresh run.py run
            ref = torch.Generator()
            ref.manual_seed(derive_seed(917, 0x22))
            self.assertTrue(torch.equal(gen.get_state(), ref.get_state()))

    def test_accepts_full_trainer_checkpoint_input(self):
        import torch
        with tempfile.TemporaryDirectory() as td:
            src, actor = self._tiny_actor(td)
            full = Path(td) / "latest.pt"
            torch.save({"actor": actor.state_dict(), "update": 123}, full)
            out = gc.make_warmstart_checkpoint(full, Path(td) / "warm.pt",
                                               seed=0, lr=1e-4)
            ck = torch.load(out, map_location="cpu", weights_only=False)
            self.assertEqual(ck["update"], 0)   # counter restarts

    def test_rejects_gru_actor(self):
        import torch
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gru.pt"
            torch.save({"arch": "gru", "state_dict": {}}, path)
            with self.assertRaises(ValueError):
                gc.make_warmstart_checkpoint(path, Path(td) / "warm.pt",
                                             seed=0, lr=1e-4)


class TestCommandBuilders(unittest.TestCase):
    def test_env_kwargs_carry_stage_grid(self):
        kw = gc.build_env_kwargs("rough4", 2)
        self.assertEqual(kw["environments"], 2)
        self.assertEqual(kw["grid"]["height_jitter"], 0.004)
        self.assertNotIn("seed", kw["grid"])
        self.assertNotIn("impulse_tolerance", kw)
        kw = gc.build_env_kwargs("flush", 4, impulse_tolerance=1e-6)
        self.assertEqual(kw["impulse_tolerance"], 1e-6)

    def test_trainer_cmd_plumbing(self):
        with tempfile.TemporaryDirectory() as td:
            args = make_args(td, workers=8, envs_per_worker=2, seed=917,
                             horizon=64, lr=7.5e-5)
            cmd = gc.trainer_cmd("rough8", Path(td) / "s", 240, args)
        self.assertIn("walk.train.run", cmd)
        i = cmd.index("--env")
        self.assertEqual(cmd[i + 1], "walk.env.grid:CubeGridDuckEnv")
        kw = json.loads(cmd[cmd.index("--env-kwargs") + 1])
        self.assertEqual(kw["grid"], gc.stage_grid("rough8"))
        self.assertEqual(kw["environments"], 2)
        self.assertEqual(cmd[cmd.index("--updates") + 1], "240")
        self.assertEqual(cmd[cmd.index("--workers") + 1], "8")
        # bare --resume (auto => <out>/latest.pt): next token is another flag
        j = cmd.index("--resume")
        self.assertTrue(cmd[j + 1].startswith("--"))

    def test_judge_cmd_plumbing(self):
        with tempfile.TemporaryDirectory() as td:
            args = make_args(td, judge_jobs=3)
            cmd = gc.judge_cmd(Path(td) / "snap.pt", "flush",
                               Path(td) / "j", args)
        self.assertIn("walk.eval.grid_acceptance", cmd)
        self.assertEqual(cmd[cmd.index("--stage") + 1], "flush")
        self.assertEqual(cmd[cmd.index("--jobs") + 1], "3")


class FakeWorld:
    """Injectable trainer/judge/clock/update-reader for stage-loop tests."""

    def __init__(self, judge_results, updates_per_chunk=20):
        self.judge_results = list(judge_results)
        self.trainer_cmds = []
        self.judged = []
        self.update = 0
        self.updates_per_chunk = updates_per_chunk
        self.t = 0.0

    def run_trainer(self, cmd):
        self.trainer_cmds.append(cmd)
        self.update += self.updates_per_chunk
        return 0

    def run_judge(self, cmd):
        self.judged.append(cmd)
        return 0 if self.judge_results.pop(0) else 1

    def clock(self):
        self.t += 60.0
        return self.t

    def read_update(self, path):
        return self.update


class TestStageAdvance(unittest.TestCase):
    def _stage_dirs(self, td):
        out = Path(td) / "cur"
        out.mkdir(parents=True, exist_ok=True)
        start = out / "warmstart.pt"
        start.write_bytes(b"fake-checkpoint")
        return out, start

    def test_advances_when_judge_passes(self):
        with tempfile.TemporaryDirectory() as td:
            out, start = self._stage_dirs(td)
            args = make_args(td, max_hours=10, poll_updates=20)
            world = FakeWorld(judge_results=[False, False, True])
            ok = gc.run_stage("flush", out / "flush", start, args,
                              out / "curriculum.jsonl",
                              run_trainer=world.run_trainer,
                              run_judge=world.run_judge,
                              clock=world.clock,
                              read_update=world.read_update)
            self.assertTrue(ok)
            self.assertEqual(len(world.judged), 3)
            self.assertEqual(len(world.trainer_cmds), 2)   # 2 failed judges
            self.assertTrue((out / "flush" / "accepted.pt").exists())
            self.assertTrue((out / "flush" / "latest.pt").exists())
            events = [json.loads(l) for l in
                      (out / "curriculum.jsonl").read_text().splitlines()]
            kinds = [e["event"] for e in events]
            self.assertEqual(kinds[0], "stage_start")
            self.assertEqual(kinds[-1], "stage_pass")
            self.assertEqual(kinds.count("judge"), 3)
            self.assertEqual(kinds.count("train_chunk"), 2)
            # trainer target updates advance by poll_updates each chunk
            targets = [int(c[c.index("--updates") + 1])
                       for c in world.trainer_cmds]
            self.assertEqual(targets, [20, 40])  # u0+20, then u20+20

    def test_zero_shot_pass_skips_training(self):
        with tempfile.TemporaryDirectory() as td:
            out, start = self._stage_dirs(td)
            args = make_args(td, max_hours=10)
            world = FakeWorld(judge_results=[True])
            ok = gc.run_stage("flush", out / "flush", start, args,
                              out / "curriculum.jsonl",
                              run_trainer=world.run_trainer,
                              run_judge=world.run_judge,
                              clock=world.clock,
                              read_update=world.read_update)
            self.assertTrue(ok)
            self.assertEqual(world.trainer_cmds, [])

    def test_max_hours_bounds_the_stage(self):
        with tempfile.TemporaryDirectory() as td:
            out, start = self._stage_dirs(td)
            args = make_args(td, max_hours=0)         # deadline already past
            world = FakeWorld(judge_results=[False])
            ok = gc.run_stage("rough4", out / "rough4", start, args,
                              out / "curriculum.jsonl",
                              run_trainer=world.run_trainer,
                              run_judge=world.run_judge,
                              clock=world.clock,
                              read_update=world.read_update)
            self.assertFalse(ok)
            self.assertEqual(world.trainer_cmds, [])   # never trained
            events = [json.loads(l) for l in
                      (out / "curriculum.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["event"], "stage_timeout")

    def test_trainer_failure_raises(self):
        with tempfile.TemporaryDirectory() as td:
            out, start = self._stage_dirs(td)
            args = make_args(td, max_hours=10)
            world = FakeWorld(judge_results=[False])
            with self.assertRaises(RuntimeError):
                gc.run_stage("flush", out / "flush", start, args,
                             out / "curriculum.jsonl",
                             run_trainer=lambda cmd: 1,
                             run_judge=world.run_judge,
                             clock=world.clock,
                             read_update=world.read_update)


class TestCurriculumChain(unittest.TestCase):
    def test_stage_chain_warm_starts_from_accepted(self):
        """Two stages, judge passes each immediately: stage 2 must start
        from stage 1's accepted.pt (real warmstart bootstrap, fake rest)."""
        import torch
        from walk.train.ppo import Actor
        with tempfile.TemporaryDirectory() as td:
            actor_path = Path(td) / "actor.pt"
            torch.save({"arch": "ff",
                        "state_dict": Actor(58, 14).state_dict()}, actor_path)
            args = make_args(td, init_actor=actor_path, max_hours=10,
                             stages="flush,rough4")
            world = FakeWorld(judge_results=[True, True])
            rc = gc.run_curriculum(args, run_trainer=world.run_trainer,
                                   run_judge=world.run_judge,
                                   clock=world.clock,
                                   read_update=world.read_update)
            out = Path(args.out)
            self.assertEqual(rc, 0)
            self.assertTrue((out / "warmstart.pt").exists())
            for stage in ("flush", "rough4"):
                self.assertTrue((out / stage / "accepted.pt").exists(), stage)
            # rough4's latest.pt was seeded from flush's accepted checkpoint
            self.assertEqual((out / "rough4" / "latest.pt").read_bytes(),
                             (out / "flush" / "accepted.pt").read_bytes())


if __name__ == "__main__":
    unittest.main()

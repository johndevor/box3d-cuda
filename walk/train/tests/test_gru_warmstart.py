"""FF -> residual-GRU warm start (walk.train.ppo.warm_start_recurrent_from_ff,
walk.train.gru_warmstart, gpu_train's cross-arch --init-actor / --resume).

Run: .venv/bin/python -B -m unittest walk.train.tests.test_gru_warmstart

Gates:
- EXACT equivalence: the warm-started RecurrentActor's deterministic action
  and (mu, std) equal the FF Actor's for random obs and ANY hidden state
  (max |diff| == 0.0), and the warm critic equals the FF critic;
- plain RecurrentActor/RecurrentCritic (no trunk) keep their original
  state-dict keys (the duck --policy gru path is unchanged);
- a generic `RecurrentActor(OBS, ACT)` grows the trunk lazily when loading
  a warm-started state dict (what the acceptance harness constructs), and
  the loaded copy is bit-identical to the source;
- gru_warmstart writes a checkpoint that gpu_train --policy gru --resume
  trains from (humanoid lane, 2 envs, 1 update, DR on, finite losses), and
  the resumed actor's trunk sizes are rebuilt from the checkpoint;
- cross-arch: --policy gru --resume <FF checkpoint> and --policy gru
  --init-actor <FF actor> both warm start instead of failing, continue /
  start the counters as documented, and their first rollout's reward lands
  in the FF specialist's own band (same policy distribution).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from walk.train import gpu_train as gt
from walk.train import gru_warmstart as gw
from walk.train import ppo

ROOT = Path(__file__).resolve().parents[3]
ACCEPTED_ACTOR = (ROOT / "evidence" / "humanoid-accepted-20260902"
                  / "actor-humanoid-accepted.pt")
ACCEPTED_CKPT = (ROOT / "runs" / "gpu" / "20260902-025432-humanoid-tree-continue"
                 / "artifacts" / "train" / "gpu-train-out" / "latest.pt")
OBS, ACT = 58, 14
DR = {"r_mass": 0.15, "r_friction": 0.3, "r_kp": 0.15, "r_damping": 0.3,
      "max_latency_steps": 2, "r_gravity": 0.5095}


def _ff_source():
    """A deterministic synthetic FF actor+critic (no evidence files needed)."""
    torch.manual_seed(3)
    actor, critic = ppo.make_nets(OBS, ACT, ppo.PPOConfig())
    with torch.no_grad():                      # make the nets non-trivial
        for p in list(actor.parameters()) + list(critic.parameters()):
            p.add_(0.05 * torch.randn_like(p))
        actor.log_std.copy_(torch.linspace(-2.0, -1.0, ACT))
    return actor, critic


class WarmStartMath(unittest.TestCase):
    def test_exact_equivalence_any_hidden_state(self):
        ff_a, ff_c = _ff_source()
        actor, critic, hidden = gw.build_warm_started(
            "humanoid", ff_a.state_dict(), ff_c.state_dict(), seed=0)
        self.assertEqual(hidden, (256, 256))
        g = torch.Generator().manual_seed(1)
        obs = torch.randn(128, OBS, generator=g)
        for h in (torch.zeros(128, actor.gru_hidden),
                  torch.randn(128, actor.gru_hidden, generator=g)):
            with torch.no_grad():
                a, _ = actor.deterministic(obs, h)
                mu, std, _ = actor.dist(obs, h)
                v, _ = critic(obs, h)
                mu_seq, std_seq, _ = actor.dist_seq(obs.view(4, 32, OBS), h[:32])
                v_seq, _ = critic.value_seq(obs.view(4, 32, OBS), h[:32])
                mu_ff, std_ff = ff_a.dist(obs)
            self.assertEqual(float((a - ff_a.deterministic(obs)).abs().max()), 0.0)
            self.assertEqual(float((mu - mu_ff).abs().max()), 0.0)
            self.assertTrue(torch.equal(std, std_ff))
            self.assertEqual(float((v - ff_c(obs)).abs().max()), 0.0)
            self.assertEqual(float((mu_seq.view(128, ACT) - mu_ff).abs().max()), 0.0)
            self.assertEqual(float((v_seq.view(128) - ff_c(obs)).abs().max()), 0.0)
        # the GRU heads' output layers are exactly zero (pure residual)
        self.assertEqual(float(actor.mu_net[-1].weight.abs().sum()), 0.0)
        self.assertEqual(float(critic.v_net[-1].weight.abs().sum()), 0.0)
        # ... and the correction pathway is trainable (nonzero head grads)
        loss = actor.dist(obs, torch.zeros(128, actor.gru_hidden))[0].pow(2).mean()
        loss.backward()
        self.assertGreater(float(actor.mu_net[-1].weight.grad.abs().sum()), 0.0)

    def test_plain_gru_state_dict_unchanged(self):
        plain_a, plain_c = ppo.make_recurrent_nets(OBS, ACT, ppo.PPOConfig())
        self.assertIsNone(plain_a.ff)
        self.assertIsNone(plain_c.ff)
        self.assertEqual(sorted(plain_a.state_dict()),
                         ["gru.bias_hh_l0", "gru.bias_ih_l0", "gru.weight_hh_l0",
                          "gru.weight_ih_l0", "log_std", "mu_net.0.bias",
                          "mu_net.0.weight", "mu_net.2.bias", "mu_net.2.weight"])
        self.assertIsNone(ppo.trunk_hidden_from_state_dict(plain_a.state_dict()))

    def test_lazy_trunk_load_matches_source(self):
        ff_a, ff_c = _ff_source()
        actor, critic, _ = gw.build_warm_started(
            "humanoid", ff_a.state_dict(), ff_c.state_dict(), seed=0)
        generic = ppo.RecurrentActor(OBS, ACT)            # harness-style
        self.assertIsNone(generic.ff)
        generic.load_state_dict(actor.state_dict())       # strict load
        self.assertIsNotNone(generic.ff)
        for k, v in actor.state_dict().items():
            self.assertTrue(torch.equal(v, generic.state_dict()[k]), k)
        gc = ppo.RecurrentCritic(OBS)
        gc.load_state_dict(critic.state_dict())
        self.assertIsNotNone(gc.ff)
        self.assertEqual(ppo.trunk_hidden_from_state_dict(actor.state_dict()),
                         (256, 256))
        with self.assertRaises(ValueError):
            ppo.warm_start_recurrent_from_ff(ppo.RecurrentActor(OBS, ACT),
                                             ff_a.state_dict())


class WarmStartTraining(unittest.TestCase):
    """Real humanoid serial lane, 2 envs, DR on (incl. gravity), tiny runs."""

    def _cfg(self, out, **over):
        base = dict(robot="humanoid", envs=2, lane_env=True, horizon=8,
                    updates=1, seed=11, device="cpu", out=str(out),
                    randomization=dict(DR), preflight_steps=0,
                    checkpoint_every=1000, torch_threads=1, quiet=True,
                    policy="gru")
        base.update(over)
        return gt.GpuTrainConfig(**base)

    def _write_ff_ckpt(self, path: Path):
        ff_a, ff_c = _ff_source()
        optimizer = torch.optim.Adam(
            list(ff_a.parameters()) + list(ff_c.parameters()), lr=3e-4)
        torch.save({"update": 5, "actor": ff_a.state_dict(),
                    "critic": ff_c.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sample_gen_state": torch.Generator().manual_seed(1).get_state(),
                    "sample_gen_device": "cpu",
                    "perm_gen_state": torch.Generator().manual_seed(2).get_state(),
                    "env_steps": 1234, "faults_total": 0, "train_wall_s": 1.0,
                    "probe_wall_s": 0.0,
                    "config": {"policy": "ff", "robot": "humanoid"}}, path)
        return ff_a

    def test_gru_warmstart_checkpoint_resumes_in_gpu_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ff_ckpt = tmp / "ff.pt"
            self._write_ff_ckpt(ff_ckpt)
            rc = gw.main(["--robot", "humanoid", "--source", str(ff_ckpt),
                          "--out", str(tmp / "gru0.pt"),
                          "--actor-out", str(tmp / "gru0-actor.pt")])
            self.assertEqual(rc, 0)
            ck = torch.load(tmp / "gru0.pt", map_location="cpu", weights_only=False)
            self.assertEqual(ck["config"]["policy"], "gru")
            self.assertEqual(ck["config"]["gru_ff_trunk"], [256, 256])
            self.assertTrue(ck["config"]["critic_warm"])
            self.assertEqual(ck["update"], 0)
            rows = [m for m in gt.train(self._cfg(tmp / "run", resume=str(tmp / "gru0.pt")))
                    if m.get("kind") == "train"]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["faults"], 0, rows[0])
            self.assertTrue(np.isfinite(rows[0]["pi_loss"]) and np.isfinite(rows[0]["v_loss"]))
            # the resumed run's actor keeps the trunk (rebuilt from the ckpt)
            arch, sd = ppo.unpack_actor_file(torch.load(
                tmp / "run" / "actor_final.pt", map_location="cpu", weights_only=False))
            self.assertEqual(arch, "gru")
            self.assertEqual(ppo.trunk_hidden_from_state_dict(sd), (256, 256))
            saved = json.loads((tmp / "run" / "config.json").read_text())
            self.assertEqual(saved["randomization"]["r_gravity"], 0.5095)

    def test_cross_arch_resume_and_init_actor_match_ff_first_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ff_ckpt = tmp / "ff.pt"
            ff_a = self._write_ff_ckpt(ff_ckpt)
            torch.save({"arch": "ff", "state_dict": ff_a.state_dict()}, tmp / "ff-actor.pt")
            # reference: the FF specialist itself, one update
            ff_rows = [m for m in gt.train(self._cfg(tmp / "ff", policy="ff",
                                                     resume=str(ff_ckpt), updates=6))
                       if m.get("kind") == "train"]
            # cross-arch resume: FF checkpoint -> residual GRU, counters continue
            x_rows = [m for m in gt.train(self._cfg(tmp / "x", resume=str(ff_ckpt),
                                                    updates=6))
                      if m.get("kind") == "train"]
            # init-actor with the FF actor FILE, fresh counters
            i_rows = [m for m in gt.train(self._cfg(tmp / "i",
                                                    init_actor=str(tmp / "ff-actor.pt"),
                                                    updates=1))
                      if m.get("kind") == "train"]
            self.assertEqual([r["update"] for r in ff_rows], [6])
            self.assertEqual([r["update"] for r in x_rows], [6])
            self.assertEqual([r["update"] for r in i_rows], [1])
            # The two warm starts hold the same policy and reseed their
            # action-noise generators from the same (seed, 0x22[, update])
            # recipe only when start_update matches; here x resumes at
            # update 5 and i starts at 0, so compare each to the FF run
            # loosely (same policy mean => rewards in the same band) and
            # check the warm starts produced a finite, non-degenerate window.
            for rows in (x_rows, i_rows):
                self.assertLess(abs(rows[0]["reward_mean"] - ff_rows[0]["reward_mean"]),
                                1.0, (rows[0], ff_rows[0]))
                self.assertGreater(rows[0]["reward_std"], 0.0)
            for rows in (ff_rows, x_rows, i_rows):
                self.assertEqual(rows[0]["faults"], 0, rows[0])
                self.assertTrue(np.isfinite(rows[0]["pi_loss"]), rows[0])
            ck = torch.load(tmp / "x" / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(ck["config"]["policy"], "gru")
            self.assertEqual(ppo.trunk_hidden_from_state_dict(ck["actor"]), (256, 256))
            self.assertEqual(ck["env_steps"], 1234 + 2 * 8)     # counters continued

    def test_arch_mismatch_still_rejected_for_ff_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plain_a, _ = ppo.make_recurrent_nets(OBS, ACT, ppo.PPOConfig())
            torch.save({"arch": "gru", "state_dict": plain_a.state_dict()}, tmp / "g.pt")
            with self.assertRaises(SystemExit):
                gt.train(self._cfg(tmp / "bad", policy="ff", init_actor=str(tmp / "g.pt")))


@unittest.skipUnless(ACCEPTED_ACTOR.is_file(), "accepted humanoid actor not present")
class AcceptedActorWarmStart(unittest.TestCase):
    def test_accepted_actor_warm_start_is_exact(self):
        sd = ppo.unpack_actor_file(torch.load(ACCEPTED_ACTOR, map_location="cpu",
                                              weights_only=False))[1]
        actor, _critic, hidden = gw.build_warm_started("humanoid", sd, None, seed=0)
        self.assertEqual(hidden, (256, 256))
        self.assertEqual(gw.verify_equivalence("humanoid", actor, sd), 0.0)
        # the acceptance harness's loader accepts the warm-started actor file
        from walk.eval.humanoid_acceptance import load_actor
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.pt"
            torch.save({"arch": "gru", "state_dict": actor.state_dict()}, p)
            arch, loaded = load_actor(str(p))
        self.assertEqual(arch, "gru")
        self.assertIsNotNone(loaded.ff)
        obs = torch.randn(8, OBS)
        with torch.no_grad():
            a1, _ = loaded.deterministic(obs, loaded.initial_state(8))
            a2, _ = actor.deterministic(obs, actor.initial_state(8))
        self.assertTrue(torch.equal(a1, a2))


if __name__ == "__main__":
    unittest.main(verbosity=2)

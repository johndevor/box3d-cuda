"""Checkpoint save/resume must reproduce training bitwise."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from walk.train.tests.test_train import LOSS_KEYS, make_cfg, train_lines
from walk.train.run import train


class TestCheckpointResume(unittest.TestCase):
    def test_resume_reproduces_identical_metrics(self):
        with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
            # Run A: 8 updates with a checkpoint at update 5.
            ma = train_lines(train(make_cfg(ta, updates=8, checkpoint_every=5)))
            ck5 = Path(ta) / "ckpt_000005.pt"
            self.assertTrue(ck5.is_file())

            # Run B: fresh process state, resume from A's update-5 checkpoint.
            mb = train_lines(train(make_cfg(tb, updates=8, checkpoint_every=5, resume=str(ck5))))
            self.assertEqual([m["update"] for m in mb], [6, 7, 8])

            tail_a = [m for m in ma if m["update"] >= 6]
            for a, b in zip(tail_a, mb):
                for k in LOSS_KEYS:
                    self.assertEqual(a[k], b[k], f"update {a['update']} field {k}: {a[k]} != {b[k]}")

            # Final weights and RNG state must match bitwise.
            cka = torch.load(Path(ta) / "ckpt_000008.pt", map_location="cpu", weights_only=False)
            ckb = torch.load(Path(tb) / "ckpt_000008.pt", map_location="cpu", weights_only=False)
            for part in ("actor", "critic"):
                for name, ta_t in cka[part].items():
                    self.assertTrue(
                        torch.equal(ta_t, ckb[part][name]),
                        f"{part}.{name} differs after resume",
                    )
            self.assertTrue(torch.equal(cka["gen_state"], ckb["gen_state"]))
            self.assertEqual(cka["env_steps"], ckb["env_steps"])


if __name__ == "__main__":
    unittest.main()

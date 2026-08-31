"""Offline regression: device policy faults must never become PPO samples."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from walk.env.contract import SolverFault
from walk.train import gpu_train


class FakeLane:
    def __init__(self, environments, **kwargs):
        self.E = environments
    def step_policy(self, action):
        diagnostics=np.zeros(self.E,dtype=[('status',np.uint32)])
        diagnostics['status'][1]=3
        return (np.zeros((self.E,58),np.float32), np.ones(self.E,np.float32),
                np.ones(self.E,bool), diagnostics)
    def state_dump(self, e):
        return {'environment':e,'step_count':7}
    def close(self):
        pass


class FaultQuarantineTests(unittest.TestCase):
    def test_policy_fault_persisted_and_raised(self):
        with tempfile.TemporaryDirectory() as tmp, patch('walk.env.cuda_lane.CudaDuckLane', FakeLane):
            env = gpu_train.LanePolicyEnv(gpu_train.GpuTrainConfig(envs=2,out=tmp))
            with self.assertRaises(SolverFault) as caught:
                env.step(np.zeros((2,14),np.float32))
            data=json.loads(Path(caught.exception.saved_problem_path).read_text())
            self.assertEqual(data['failed_environments'],[1])
            self.assertFalse(data['rollout_accepted'])
            self.assertFalse(data['exact_replay_available'])
            self.assertEqual(env.fault_count,1)

    def test_fault_does_not_produce_rollout_or_episode_samples(self):
        with tempfile.TemporaryDirectory() as tmp, patch('walk.env.cuda_lane.CudaDuckLane', FakeLane):
            env = gpu_train.LanePolicyEnv(gpu_train.GpuTrainConfig(envs=2,out=tmp))
            actor,critic=gpu_train.make_nets(58,14,gpu_train.PPOConfig())
            returns=np.zeros(2);lengths=np.zeros(2,dtype=np.int64)
            with self.assertRaises(SolverFault):
                gpu_train.collect_rollout(env,actor,critic,np.zeros((2,58),np.float32),
                    1,torch.Generator().manual_seed(1),torch.device('cpu'),returns,lengths)
            np.testing.assert_array_equal(returns,0)
            np.testing.assert_array_equal(lengths,0)


if __name__=='__main__':unittest.main()

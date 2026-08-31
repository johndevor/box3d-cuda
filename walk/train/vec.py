"""Multiprocess vectorized environment over DuckEnvBatch shards.

Each of N worker processes constructs its OWN env shard from a dotted factory
spec ("walk.env.contract:StubEnv" + kwargs), so native ctypes state never
crosses a process boundary. Observations/actions/rewards/dones are exchanged
through POSIX shared memory (multiprocessing.shared_memory); the per-step
protocol is barrier-free: the learner writes actions into shared memory, sends
one tiny message down each worker's pipe, and drains each worker's tiny reply.

Worker kwargs conveniences (applied by VecEnv before spawning):
  - env_kwargs[env_count_key] is overwritten with the shard size
    (total_envs // workers), so `--env-kwargs '{"environments":16}'` with
    `--workers 12 --total-envs 192` gives 16 envs per worker.
  - any top-level kwarg whose value is the string "$WORKER" is replaced by the
    worker index (useful for test envs / per-shard variation).

SolverFault from any worker: the worker persists nothing itself (the env
already saved the failing inputs per the contract), replies with the fault
info, auto-resets its WHOLE shard with a deterministic seed, and keeps
serving. The learner surfaces the fault in step() info so the trainer can
poison that shard's partial rollout and log the artifact path.

This module deliberately does not import torch: workers only need numpy.
Worker processes are started with thread-limiting env vars (OMP etc.) set to 1
and call torch.set_num_threads(1) if torch happens to be loaded by the env.
"""
from __future__ import annotations

import dataclasses
import importlib
import multiprocessing as mp
import os
import sys
import traceback
from multiprocessing import shared_memory

import numpy as np

from walk.env.contract import ACT as CONTRACT_ACT, OBS as CONTRACT_OBS, SolverFault

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def load_factory(spec: str):
    """Resolve 'package.module:Attr' to the attribute."""
    mod, sep, attr = spec.partition(":")
    if not sep or not mod or not attr:
        raise ValueError(f"env spec must look like 'package.module:Factory', got {spec!r}")
    return getattr(importlib.import_module(mod), attr)


def derive_seed(*parts: int) -> int:
    """Deterministic, well-mixed 31-bit seed from integer parts."""
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0] & 0x7FFFFFFF)


@dataclasses.dataclass
class FaultRecord:
    worker: int
    env_index: int  # global env index (worker offset + env's index in shard)
    saved_problem_path: str
    message: str


def _limit_worker_threads() -> None:
    if "torch" in sys.modules:
        try:
            sys.modules["torch"].set_num_threads(1)
        except Exception:
            pass


def _disable_child_shm_tracking() -> None:
    """The parent owns every segment; don't let workers (double-)track them.

    Python < 3.13 has no SharedMemory(track=False), and register/unregister
    round-trips from multiple children race in the shared resource tracker,
    spewing KeyError tracebacks at exit.
    """
    from multiprocessing import resource_tracker

    if getattr(resource_tracker.register, "_dgw_patched", False):
        return
    orig = resource_tracker.register

    def _register(name, rtype):
        if rtype == "shared_memory":
            return
        orig(name, rtype)

    _register._dgw_patched = True  # type: ignore[attr-defined]
    resource_tracker.register = _register


def _attach_shm(name: str) -> shared_memory.SharedMemory:
    try:
        return shared_memory.SharedMemory(name=name, track=False)  # py >= 3.13
    except TypeError:
        _disable_child_shm_tracking()
        return shared_memory.SharedMemory(name=name)


def _worker_main(wid, spec, env_kwargs, shm_names, n_total, obs_dim, act_dim, lo, hi, conn, base_seed):
    _limit_worker_threads()
    np.random.seed(derive_seed(base_seed, 0xA11CE, wid))  # legacy global, for env code that uses it
    shms = {k: _attach_shm(v) for k, v in shm_names.items()}
    n = hi - lo
    obs_buf = np.ndarray((n_total, obs_dim), np.float32, buffer=shms["obs"].buf)[lo:hi]
    act_buf = np.ndarray((n_total, act_dim), np.float32, buffer=shms["act"].buf)[lo:hi]
    rew_buf = np.ndarray((n_total,), np.float32, buffer=shms["rew"].buf)[lo:hi]
    done_buf = np.ndarray((n_total,), np.uint8, buffer=shms["done"].buf)[lo:hi]
    env = None
    try:
        env = load_factory(spec)(**env_kwargs)
        _limit_worker_threads()  # env import may have pulled in torch
        if env.E != n:
            raise RuntimeError(f"worker {wid}: env.E={env.E} != shard size {n}")
        if env.OBS != obs_dim or env.ACT != act_dim:
            raise RuntimeError(
                f"worker {wid}: env OBS/ACT=({env.OBS},{env.ACT}) != shm ({obs_dim},{act_dim}); "
                "pass obs_dim= to VecEnv (or --obs-dim to run.py)"
            )
        ep_ret = np.zeros(n, np.float64)
        ep_len = np.zeros(n, np.int64)
        fault_count = 0
        conn.send(("ready", None))
        while True:
            msg = conn.recv()
            cmd = msg[0]
            if cmd == "step":
                try:
                    obs, rew, done, _info = env.step(act_buf.copy())
                except SolverFault as f:
                    fault_count += 1
                    reseed = derive_seed(base_seed, 0xFA017, wid, fault_count)
                    obs = env.reset(seed=reseed)
                    obs_buf[:] = obs
                    rew_buf[:] = 0.0
                    done_buf[:] = 0
                    ep_ret[:] = 0.0
                    ep_len[:] = 0
                    conn.send(("fault", (lo + int(f.env_index), str(f.saved_problem_path), str(f))))
                    continue
                done = np.asarray(done, bool)
                ep_ret += rew
                ep_len += 1
                episodes = []
                if done.any():
                    for i in np.nonzero(done)[0]:
                        episodes.append((float(ep_ret[i]), int(ep_len[i])))
                    ep_ret[done] = 0.0
                    ep_len[done] = 0
                    # contract: envs stay terminal until reset(mask); reset now so
                    # the next policy step sees fresh-episode observations.
                    reset_obs = env.reset(mask=done)
                    obs = np.array(obs, np.float32, copy=True)
                    obs[done] = reset_obs[done]
                obs_buf[:] = obs
                rew_buf[:] = rew
                done_buf[:] = done
                conn.send(("ok", episodes))
            elif cmd == "reset":
                obs_buf[:] = env.reset(seed=int(msg[1]))
                rew_buf[:] = 0.0
                done_buf[:] = 0
                ep_ret[:] = 0.0
                ep_len[:] = 0
                conn.send(("ok", None))
            elif cmd == "close":
                conn.send(("ok", None))
                break
            else:
                raise RuntimeError(f"worker {wid}: unknown command {cmd!r}")
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        for s in shms.values():
            try:
                s.close()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


class VecEnv:
    """N worker processes, each owning total_envs // workers environments."""

    def __init__(
        self,
        spec: str,
        env_kwargs: dict | None = None,
        workers: int = 1,
        total_envs: int | None = None,
        seed: int = 0,
        env_count_key: str = "environments",
        obs_dim: int | None = None,
        act_dim: int | None = None,
        start_method: str = "spawn",
    ):
        env_kwargs = dict(env_kwargs or {})
        workers = int(workers)
        if workers < 1:
            raise ValueError("workers must be >= 1")
        if total_envs is None:
            total_envs = int(env_kwargs.get(env_count_key, 16)) * workers
        total_envs = int(total_envs)
        if total_envs % workers != 0:
            raise ValueError(f"total_envs={total_envs} must be divisible by workers={workers}")
        shard = total_envs // workers

        factory = load_factory(spec)  # early validation in the learner too
        self.obs_dim = int(obs_dim if obs_dim is not None else getattr(factory, "OBS", CONTRACT_OBS))
        self.act_dim = int(act_dim if act_dim is not None else getattr(factory, "ACT", CONTRACT_ACT))
        self.spec = spec
        self.workers = workers
        self.total_envs = total_envs
        self.shard = shard
        self.seed = int(seed)
        self.slices = [slice(w * shard, (w + 1) * shard) for w in range(workers)]
        self.fault_count = 0
        self._closed = False

        f4 = np.dtype(np.float32).itemsize
        self._shms = {
            "obs": shared_memory.SharedMemory(create=True, size=total_envs * self.obs_dim * f4),
            "act": shared_memory.SharedMemory(create=True, size=total_envs * self.act_dim * f4),
            "rew": shared_memory.SharedMemory(create=True, size=total_envs * f4),
            "done": shared_memory.SharedMemory(create=True, size=total_envs),
        }
        self._obs = np.ndarray((total_envs, self.obs_dim), np.float32, buffer=self._shms["obs"].buf)
        self._act = np.ndarray((total_envs, self.act_dim), np.float32, buffer=self._shms["act"].buf)
        self._rew = np.ndarray((total_envs,), np.float32, buffer=self._shms["rew"].buf)
        self._done = np.ndarray((total_envs,), np.uint8, buffer=self._shms["done"].buf)

        ctx = mp.get_context(start_method)
        shm_names = {k: s.name for k, s in self._shms.items()}
        self._procs: list = []
        self._conns: list = []
        saved = {k: os.environ.get(k) for k in _THREAD_ENV_VARS}
        for k in _THREAD_ENV_VARS:
            os.environ[k] = "1"  # inherited by workers at spawn; keeps their BLAS single-threaded
        try:
            for w in range(workers):
                kw = {k: (w if v == "$WORKER" else v) for k, v in env_kwargs.items()}
                kw[env_count_key] = shard
                parent, child = ctx.Pipe()
                p = ctx.Process(
                    target=_worker_main,
                    args=(w, spec, kw, shm_names, total_envs, self.obs_dim, self.act_dim,
                          w * shard, (w + 1) * shard, child, self.seed),
                    daemon=True,
                    name=f"vecenv-w{w}",
                )
                p.start()
                child.close()
                self._procs.append(p)
                self._conns.append(parent)
            for w, c in enumerate(self._conns):
                kind, payload = self._recv(w, c)
                if kind != "ready":
                    raise RuntimeError(f"worker {w}: unexpected startup reply {kind!r}")
        except Exception:
            self.close()
            raise
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    # -- protocol -----------------------------------------------------------

    def _recv(self, w: int, conn):
        try:
            kind, payload = conn.recv()
        except EOFError as e:
            raise RuntimeError(f"vec worker {w} died unexpectedly (pipe closed)") from e
        if kind == "error":
            raise RuntimeError(f"vec worker {w} raised:\n{payload}")
        return kind, payload

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset all shards with per-worker seeds derived from `seed`."""
        base = self.seed if seed is None else int(seed)
        for w, c in enumerate(self._conns):
            c.send(("reset", derive_seed(base, 0x5EED, w)))
        for w, c in enumerate(self._conns):
            self._recv(w, c)
        return self._obs.copy()

    def step(self, actions: np.ndarray):
        """Step all shards. Returns (obs, rew, done, info).

        info = {"faults": [FaultRecord ...], "episodes": [(return, length) ...]}
        A faulted shard returns its post-reset observations with rew=0/done=False;
        the caller must treat that shard's window as poisoned.
        """
        a = np.asarray(actions, np.float32)
        if a.shape != (self.total_envs, self.act_dim):
            raise ValueError(f"actions shape {a.shape} != {(self.total_envs, self.act_dim)}")
        self._act[:] = a
        for c in self._conns:
            c.send(("step",))
        faults: list[FaultRecord] = []
        episodes: list[tuple[float, int]] = []
        for w, c in enumerate(self._conns):
            kind, payload = self._recv(w, c)
            if kind == "fault":
                gi, path, msg = payload
                faults.append(FaultRecord(worker=w, env_index=int(gi), saved_problem_path=path, message=msg))
                self.fault_count += 1
            else:
                episodes.extend(payload)
        return (
            self._obs.copy(),
            self._rew.copy(),
            self._done.astype(bool),
            {"faults": faults, "episodes": episodes},
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for c in getattr(self, "_conns", []):
            try:
                c.send(("close",))
            except Exception:
                pass
        for p in getattr(self, "_procs", []):
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        for c in getattr(self, "_conns", []):
            try:
                c.close()
            except Exception:
                pass
        for s in getattr(self, "_shms", {}).values():
            try:
                s.close()
            except Exception:
                pass
            try:
                s.unlink()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

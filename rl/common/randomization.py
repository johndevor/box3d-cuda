"""The randomization spec format shared by every skill.

A spec is an ordered mapping name -> entry. An entry is a distribution or a plain setting:

  U(lo, hi)        uniform                      {"dist": "uniform", "range": [lo, hi]}
  LogU(lo, hi)     log-uniform                  {"dist": "loguniform", "range": [lo, hi]}
  Bern(p)          1.0 with probability p       {"dist": "bernoulli", "p": p}
  IntU(lo, hi)     integer in [lo, hi)          {"dist": "int", "range": [lo, hi]}
  a number/str/... a fixed setting              {"value": v}

`Spec.draw(name, g, n, device)` samples n values with the entry's distribution (the arithmetic is the one the skills
always used, so a seed gives the same worlds as before); `spec[name]` is the raw value (a (lo, hi) tuple for ranges,
the probability for Bern, the setting otherwise) for code that reads bounds; `spec.to_json()` goes into the manifest;
`spec.override(json)` takes the same JSON form, or a bare [lo, hi] / number that keeps the entry's distribution.
Pinning an entry to one value (lo == hi) is how the sim-match turns randomization off.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class U:
    lo: float
    hi: float
    kind = "uniform"

    def raw(self):
        return (self.lo, self.hi)

    def draw(self, g, n, device):
        return self.lo + (self.hi - self.lo) * torch.rand(n, generator=g, device=device)


@dataclass(frozen=True)
class LogU(U):
    kind = "loguniform"

    def draw(self, g, n, device):
        lo, hi = self.lo, self.hi
        return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(n, generator=g, device=device))


@dataclass(frozen=True)
class Bern:
    p: float
    kind = "bernoulli"

    def raw(self):
        return self.p

    def draw(self, g, n, device):
        return (torch.rand(n, generator=g, device=device) < self.p).float()


@dataclass(frozen=True)
class IntU:
    lo: int
    hi: int
    kind = "int"

    def raw(self):
        return (self.lo, self.hi)

    def draw(self, g, n, device):
        return torch.randint(self.lo, self.hi, (n,), generator=g, device=device)


DISTS = {"uniform": U, "loguniform": LogU, "bernoulli": Bern, "int": IntU}


def _entry_json(e):
    if isinstance(e, Bern):
        return {"dist": e.kind, "p": e.p}
    if isinstance(e, (U, IntU)):
        return {"dist": e.kind, "range": [e.lo, e.hi]}
    return {"value": list(e) if isinstance(e, tuple) else e}


def _entry_from(old, v):
    if isinstance(v, dict) and "dist" in v:
        cls = DISTS[v["dist"]]
        return cls(v["p"]) if cls is Bern else cls(*v["range"])
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    if isinstance(old, Bern):
        return Bern(float(v))
    if isinstance(old, (U, IntU)) and isinstance(v, (list, tuple)):
        return type(old)(*v)
    if isinstance(old, (U, IntU)) and isinstance(v, (int, float)):      # pinned
        return type(old)(v, v + (1 if isinstance(old, IntU) else 0))
    return tuple(v) if isinstance(v, list) else v


class Spec:
    def __init__(self, entries: dict):
        self.e = dict(entries)

    def __getitem__(self, k):
        v = self.e[k]
        return v.raw() if isinstance(v, (U, IntU, Bern)) else v

    def __setitem__(self, k, v):
        self.e[k] = _entry_from(self.e.get(k), v)

    def __contains__(self, k):
        return k in self.e

    def get(self, k, d=None):
        return self[k] if k in self.e else d

    def keys(self):
        return self.e.keys()

    def draw(self, k, g, n, device):
        e = self.e[k]
        if not hasattr(e, "draw"):
            raise TypeError(f"randomization entry {k} is a fixed setting ({e!r}), not a distribution")
        return e.draw(g, n, device)

    def override(self, over: dict | None):
        s = Spec(self.e)
        for k, v in (over or {}).items():
            if k not in s.e:
                raise KeyError(f"unknown randomization entry {k!r}")
            s.e[k] = _entry_from(s.e[k], v)
        return s

    def to_json(self):
        return {k: _entry_json(v) for k, v in self.e.items()}

    def raw_dict(self):
        """name -> raw value (the dict form skills used before the spec format)."""
        return {k: self[k] for k in self.e}

    def __repr__(self):
        return f"Spec({len(self.e)} entries)"


def as_spec(x) -> Spec:
    return x if isinstance(x, Spec) else Spec(x)

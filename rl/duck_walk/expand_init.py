"""Warm start for the v2 interface (foot switches + gait clock): a v1 checkpoint's networks with the new observation
columns added at zero weight (the policy starts as the v1 policy and learns to use the new inputs).
  python -m rl.duck_walk.expand_init <v1 ckpt> <out ckpt>
"""
import sys

import torch

from .env import FRAME, HISTORY
from .train import ActorCritic

src, out = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu")
cfg = ck["cfg"]
old_f, new_f, priv = FRAME, FRAME + 4, cfg["priv"]
old = ActorCritic(old_f * HISTORY, priv, cfg["act"]); old.load_state_dict(ck["model"])
new = ActorCritic(new_f * HISTORY, priv, cfg["act"])
idx = torch.cat([torch.arange(old_f) + h * new_f for h in range(HISTORY)])        # where each old obs column goes
with torch.no_grad():
    new.load_state_dict({k: v for k, v in old.state_dict().items() if not (k.startswith("actor.0") or k.startswith("critic.0") or "norm" in k)}, strict=False)
    a0, c0 = new.actor[0], new.critic[0]
    a0.weight.zero_(); a0.weight[:, idx] = old.actor[0].weight; a0.bias.copy_(old.actor[0].bias)
    cidx = torch.cat([idx, torch.arange(priv) + new_f * HISTORY])
    c0.weight.zero_(); c0.weight[:, cidx] = old.critic[0].weight; c0.bias.copy_(old.critic[0].bias)
    for nn_, on in ((new.obs_norm, old.obs_norm), (new.crit_norm, old.crit_norm)):
        sel = idx if nn_ is new.obs_norm else cidx
        nn_.mean.zero_(); nn_.var.fill_(1.0); nn_.mean[sel] = on.mean; nn_.var[sel] = on.var; nn_.count.copy_(on.count)
    new.log_std.copy_(old.log_std)
cfg = dict(cfg, obs=new_f * HISTORY)
torch.save({"model": new.state_dict(), "steps": ck.get("steps"), "cfg": cfg}, out)
print("expanded", src, "->", out, "obs", old_f * HISTORY, "->", new_f * HISTORY)

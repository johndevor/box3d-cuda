#!/usr/bin/env python3
"""Export an orbitable 3D replay of a humanoid actor to a single HTML file.

Runs the actor closed-loop on the CPU serial lane (E=1), records every body
pose at each 20 ms policy step, and writes a self-contained three.js viewer
(orbit camera, play/pause/scrub, per-clip selector). Box geometry straight
from the active lowering — what the solver simulates is what you see.

  .venv/bin/python -B scripts/export_humanoid_view.py \
      --clip "BC clone:humanoid/bc_init.pt" \
      --clip "tree leg u59:runs/gpu/20260902-000009-humanoid-tree/artifacts/train/gpu-train-out/actor_final.pt" \
      --command 0.75 --seconds 8 --out evidence/humanoid-progress-20260902/replay.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))

import h1_lowering as h1  # noqa: E402
from walk.env.humanoid_flat import FlatFloorHumanoidEnv  # noqa: E402
from walk.train.ppo import Actor, RecurrentActor, unpack_actor_file  # noqa: E402


def record_clip(actor_path: str, command: float, seconds: float, seed: int):
    raw = torch.load(actor_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "actor" in raw:
        raw = raw["actor"]
    arch, sd = unpack_actor_file(raw)
    from walk.env import humanoid_flat as hf
    actor = (RecurrentActor(hf.OBS, hf.ACT) if arch == "gru"
             else Actor(hf.OBS, hf.ACT))
    actor.load_state_dict(sd)
    actor.eval()

    env = FlatFloorHumanoidEnv(environments=1, seed=seed)
    obs = env.reset()
    env.set_command(np.full(1, command, np.float64))
    frames, fell_at = [], None
    h = None
    with torch.no_grad():
        for t in range(int(seconds / 0.02)):
            state = env._lane.read()
            frames.append(np.round(state.body_state[0, :, :7], 4).tolist())
            o = torch.from_numpy(np.ascontiguousarray(obs))
            if arch == "gru":
                if h is None:
                    h = actor.initial_state(1)
                a, h = actor.deterministic(o, h)
                a = a.numpy()
            else:
                a = actor.deterministic(o).numpy()
            obs, _r, done, _ = env.step(a)
            if done.any():
                fell_at = round((t + 1) * 0.02, 2)
                break
    env.close()
    return frames, fell_at


HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Humanoid H1 — progress replay</title>
<style>body{margin:0;background:#101318;color:#dfe5ec;font:13px system-ui}
#hud{position:fixed;top:10px;left:10px;background:#1a1f27cc;padding:10px 14px;border-radius:8px}
#hud select,#hud button,#hud input{margin:2px 4px 2px 0}</style></head><body>
<div id="hud"><b>Humanoid H1</b> — drag to orbit, wheel to zoom<br>
<select id="clip"></select><button id="play">pause</button>
<input id="scrub" type="range" min="0" value="0" style="width:220px">
<span id="status"></span></div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
"three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const DATA = __DATA__;
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x101318);
const cam = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.01, 100);
cam.position.set(2.6, -2.6, 1.9); cam.up.set(0,0,1);
const rnd = new THREE.WebGLRenderer({antialias:true});
rnd.setSize(innerWidth, innerHeight); document.body.appendChild(rnd.domElement);
const ctl = new OrbitControls(cam, rnd.domElement); ctl.target.set(0,0,0.8);
scene.add(new THREE.HemisphereLight(0xcfd8e3, 0x33383f, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(3,2,6); scene.add(sun);
const grid = new THREE.GridHelper(20, 40, 0x3a4250, 0x232a34);
grid.rotation.x = Math.PI/2; scene.add(grid);
const mats = {foot:0x64b5f6, pelvis:0xffb74d, def:0x90a4ae};
const boxes = DATA.bodies.map((b,i)=>{
  if (b.name==='floor') return null;
  const m = new THREE.Mesh(new THREE.BoxGeometry(b.half[0]*2,b.half[1]*2,b.half[2]*2),
    new THREE.MeshStandardMaterial({color: b.name.includes('foot')?mats.foot:
      (b.name==='pelvis'?mats.pelvis:mats.def), metalness:.1, roughness:.65}));
  scene.add(m); return m;});
let clip = 0, frame = 0, playing = true;
const sel = document.getElementById('clip'), scrub = document.getElementById('scrub'),
      status = document.getElementById('status');
DATA.clips.forEach((c,i)=>{const o=document.createElement('option');o.value=i;
  o.textContent=c.name+(c.fell_at?` (falls @ ${c.fell_at}s)`:' (survives)');sel.appendChild(o);});
sel.onchange = e=>{clip=+e.target.value; frame=0; scrub.max=DATA.clips[clip].frames.length-1;};
scrub.max = DATA.clips[0].frames.length-1;
scrub.oninput = e=>{frame=+e.target.value; playing=false; document.getElementById('play').textContent='play';};
document.getElementById('play').onclick = e=>{playing=!playing;
  e.target.textContent = playing?'pause':'play';};
function show(f){const fr = DATA.clips[clip].frames[f];
  fr.forEach((s,i)=>{const m=boxes[i]; if(!m)return;
    m.position.set(s[0],s[1],s[2]); m.quaternion.set(s[3],s[4],s[5],s[6]);});
  status.textContent = ` t=${(f*0.02).toFixed(2)}s  cmd ${DATA.clips[clip].command} m/s`;
  scrub.value = f;}
let last = 0;
function loop(ts){requestAnimationFrame(loop);
  if (playing && ts-last > 40){last=ts;
    frame = (frame+1) % DATA.clips[clip].frames.length;}
  show(frame); ctl.update(); rnd.render(scene, cam);}
loop(0);
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;
  cam.updateProjectionMatrix(); rnd.setSize(innerWidth,innerHeight);});
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", required=True,
                    metavar="NAME:ACTOR_PATH")
    ap.add_argument("--command", type=float, default=0.75)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default="evidence/humanoid-progress/replay.html")
    a = ap.parse_args()

    clips = []
    for spec in a.clip:
        name, _, path = spec.partition(":")
        frames, fell_at = record_clip(path, a.command, a.seconds, a.seed)
        clips.append({"name": name, "command": a.command,
                      "fell_at": fell_at, "frames": frames})
        print(f"{name}: {len(frames)} frames"
              + (f", falls @ {fell_at}s" if fell_at else ", survives"))

    bodies = [{"name": n, "half": list(b[2])}
              for n, b in zip(h1.BODY_NAMES, h1.BODIES)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__DATA__", json.dumps(
        {"bodies": bodies, "clips": clips})))
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export a self-contained CAD-rendered walking replay for a trained policy.

Runs the policy on FlatFloorDuckEnv, records root+joint trajectories per
native tick, computes per-body world poses through the sealed Open Duck FK
helpers (source STL CAD), and writes one self-contained HTML viewer.

Usage:
  .venv/bin/python -B scripts/export_walk_view.py \
      --checkpoint runs/flat-003/latest.pt --command 0.15 \
      --library build/libintegrated_duck-pinned-97c3d37.dylib \
      --asset-root /absolute/path/to/historical-cad-bundle \
      --out runs/walk-view
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from walk.env.flat import FlatFloorDuckEnv  # noqa: E402
from walk.train.ppo import Actor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--command", type=float, default=0.15)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--every", type=int, default=10, help="keep every Nth 2 ms tick")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--library", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--asset-root", type=Path, required=True,
                    help="historical CAD bundle root containing scripts/ and evidence/")
    ap.add_argument("--grid", default=None,
                    help="JSON grid spec: render duck on a cube grid, e.g. "
                         "'{\"nx\":8,\"nz\":8,\"cube_size\":0.06,\"spacing\":0.06,\"height_jitter\":0.005}'")
    a = ap.parse_args()

    old = a.asset_root.resolve()
    view_json = old / "evidence/open-duck-zero-hold-view-v1/open-duck-zero-hold-view.json"
    xml = old / "evidence/open-duck-zero-hold-cpu-v1/model/open_duck_mini_v2.xml"
    helper = old / "scripts/export_open_duck_recorded_view.py"
    for path in [view_json, xml, helper]:
        if not path.is_file():ap.error(f"missing CAD input: {path}")
    if a.grid and json.loads(a.grid).get('dynamic', False):
        ap.error('dynamic cubes require per-frame poses; this exporter supports static grids only')
    sys.path.insert(0, str(old))
    spec = importlib.util.spec_from_file_location("duck_cad_export_helper", helper)
    cad = importlib.util.module_from_spec(spec);spec.loader.exec_module(cad)
    fk_bodies, _, _ = cad.load_model(xml)
    base_view = json.loads(view_json.read_text())

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    actor = Actor(58, 14)
    actor.load_state_dict(ck["actor"])
    actor.eval()

    @torch.no_grad()
    def policy(obs):
        return actor.deterministic(torch.from_numpy(np.ascontiguousarray(obs))).numpy()

    if a.grid:
        from walk.env.grid import CubeGridDuckEnv
        env = CubeGridDuckEnv(environments=1, seed=a.seed, perturbation_rad=0.0,
                              grid=json.loads(a.grid), library_path=a.library)
    else:
        env = FlatFloorDuckEnv(environments=1, seed=a.seed, perturbation_rad=0.0,
                               library_path=a.library)
    ticks = []

    def on_tick(state):
        ticks.append((state.q[0].copy(), state.foot_contact[0].copy(),
                      float(state.time[0])))

    obs = env.reset(seed=a.seed)
    obs = env.set_command(a.command)
    steps = int(round(a.seconds / 0.02))
    for _ in range(steps):
        obs, _, done, _ = env.step(policy(obs), on_tick=on_tick)
        if done.all():
            break

    frames = []
    kept = ticks[:: a.every]
    for q, contact, t in kept:
        # lane root quat is xyzw at q[3:7]; FK wants pos + wxyz
        base_pose = [float(q[0]), float(q[1]), float(q[2]),
                     float(q[6]), float(q[3]), float(q[4]), float(q[5])]
        joint_q = [float(x) for x in q[7:21]]
        poses = cad.forward_kinematics(fk_bodies, {"base_pose": base_pose,
                                                   "joint_q": joint_q})
        gvec = np.array([2 * (q[3] * q[5] - q[6] * q[4]),
                         2 * (q[4] * q[5] + q[6] * q[3]),
                         1 - 2 * (q[3] ** 2 + q[4] ** 2)])
        tilt = float(np.degrees(np.arccos(np.clip(gvec[2], -1, 1))))
        frames.append({"time_s": round(t, 4),
                       "poses": [[[round(v, 5) for v in p[0]],
                                  [round(v, 6) for v in p[1]]] for p in poses],
                       "contacts": [bool(contact[0]), bool(contact[1])],
                       "root_x": round(float(q[0]), 5), "tilt_deg": round(tilt, 2)})

    # sanity: FK trunk position must match the recorded root position
    f0 = frames[0]["poses"][1][0]
    q0 = kept[0][0]
    drift = float(np.linalg.norm(np.array(f0) - q0[:3]))
    if drift > 0.06:
        raise SystemExit(f"FK/root mismatch {drift:.3f} m - convention error")

    extra_geometry, extra_body = [], None
    if a.grid:
        cubes = np.asarray(env._lane.state_dump(0)["cube_pose"], float)
        half = float(json.loads(a.grid).get("cube_size", 0.06)) / 2.0
        cube_body_index = 16  # one synthetic identity-pose body for all cubes
        extra_body = {"name": "cube_grid", "parent": -1}
        # 12 triangles per cube, world-space vertices (static grid: pose fixed)
        F = [(0,1,2),(0,2,3),(4,6,5),(4,7,6),(0,4,5),(0,5,1),
             (3,2,6),(3,6,7),(1,5,6),(1,6,2),(0,3,7),(0,7,4)]
        for cx, cy, cz in cubes[:, :3]:
            V = [[cx+sx*half, cy+sy*half, cz+sz*half]
                 for sz in (-1, 1) for sx, sy in ((-1,-1),(1,-1),(1,1),(-1,1))]
            tris = [[[round(v, 5) for v in V[i]] for i in f] for f in F]
            extra_geometry.append({"name": "cube", "body": cube_body_index,
                                   "mesh": "cube", "collision": True,
                                   "rgba": [0.55, 0.44, 0.30, 1.0],
                                   "triangles": tris})
        identity = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        for f in frames:
            f["poses"].append(identity)

    data = {"schema": "duckgridwalk.walk-cad-view/v1",
            "checkpoint": str(a.checkpoint), "update": ck.get("update"),
            "command_mps": a.command, "frame_dt_s": 0.002 * a.every,
            "physics_backend": "Box3D cube-grid CPU lane (dwv1/civ1)" if a.grid else "Box3D integrated duck CPU lane (idv1/civ1)",
            "asset_notice": base_view["asset_notice"],
            "bodies": base_view["bodies"] + ([extra_body] if extra_body else []),
            "geometry": base_view["geometry"] + extra_geometry,
            "grid": json.loads(a.grid) if a.grid else None,
            "frames": frames}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    tag = "-grid" if a.grid else ""
    name = f"walk-u{ck.get('update')}-cmd{a.command:.2f}{tag}.html"
    (out / name).write_text(html)
    env.close()
    print(out / name)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Open Duck walking replay</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{margin:0;background:#101014;color:#e8e8e6;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:14px 20px 6px}
h1{font-size:16px;margin:0;font-weight:640}
p{margin:2px 0 0;font-size:12px;color:#9a9a94}
.wrap{padding:8px 20px 20px;max-width:1060px}
canvas{width:100%;background:#16161c;border:1px solid #2a2a30;border-radius:10px;display:block}
.bar{display:flex;gap:10px;align-items:center;margin-top:10px;font-size:12.5px}
button,select{background:#22222a;color:#e8e8e6;border:1px solid #34343c;border-radius:7px;padding:5px 12px;font-size:12.5px;cursor:pointer}
input[type=range]{flex:1}
.cap{margin-top:6px;font-size:12px;color:#9a9a94;font-variant-numeric:tabular-nums}
.chip{display:inline-block;padding:1px 8px;border-radius:9px;font-size:11px;margin-left:6px;border:1px solid #34343c}
.on{background:#123a22;color:#7ee2a2;border-color:#1d5c36}
.off{background:#222;color:#777}
</style></head><body>
<header><h1>Open Duck — learned policy replay (source CAD, recorded physics states)</h1>
<p id="sub"></p></header>
<div class="wrap">
<canvas id="c" width="2040" height="900"></canvas>
<div class="bar">
<button id="play">Pause</button>
<select id="speed"><option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option></select>
<input type="range" id="scrub" min="0" value="0">
<label style="font-size:12px;color:#9a9a94"><input type="checkbox" id="follow" checked> follow</label>
<span style="font-size:11.5px;color:#6d6d75">drag to orbit &middot; scroll to zoom</span>
</div>
<div class="cap" id="cap"></div>
</div>
<script>
const D = __DATA__;
document.getElementById('sub').textContent =
 `checkpoint update ${D.update} · command +${D.command_mps.toFixed(2)} m/s · ${D.frames.length} recorded frames @ ${(D.frame_dt_s*1000).toFixed(0)} ms · ${D.physics_backend} · viewer advances no physics`;
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
const scrub=document.getElementById('scrub');scrub.max=D.frames.length-1;
let frame=0,playing=true,acc=0,last=performance.now();
// orbit camera, z-up: drag to orbit, wheel to zoom, side view by default
let az=-Math.PI/2, elev=0.28, zoom=1650;
let basis;
function updateBasis(){
  const d=[Math.cos(elev)*Math.cos(az),Math.cos(elev)*Math.sin(az),Math.sin(elev)];
  const f=[-d[0],-d[1],-d[2]];                        // view direction
  let r=[f[1],-f[0],0]; const rl=Math.hypot(r[0],r[1])||1; r=[r[0]/rl,r[1]/rl,0];
  const u=[r[1]*f[2]-r[2]*f[1], r[2]*f[0]-r[0]*f[2], r[0]*f[1]-r[1]*f[0]];
  basis={r,u,f};
}
updateBasis();
function project(p,camx){
  const x=p[0]-camx,y=p[1],z=p[2]-0.12;               // target mid-duck height
  const {r,u,f}=basis;
  return [1020+(x*r[0]+y*r[1]+z*r[2])*zoom,
          620-(x*u[0]+y*u[1]+z*u[2])*zoom,
          x*f[0]+y*f[1]+z*f[2]];
}
cv.addEventListener('pointerdown',e=>{
  cv.setPointerCapture(e.pointerId);
  let px=e.clientX,py=e.clientY;
  const move=ev=>{az-=(ev.clientX-px)*0.008; elev=Math.min(1.45,Math.max(-0.1,elev+(ev.clientY-py)*0.006));
                  px=ev.clientX;py=ev.clientY;updateBasis();};
  const up=()=>{cv.removeEventListener('pointermove',move);cv.removeEventListener('pointerup',up);};
  cv.addEventListener('pointermove',move);cv.addEventListener('pointerup',up);
});
cv.addEventListener('wheel',e=>{e.preventDefault();
  zoom=Math.min(5000,Math.max(400,zoom*Math.exp(-e.deltaY*0.001)));},{passive:false});
function rot(q,v){ // wxyz active
  const [w,x,y,z]=q,[vx,vy,vz]=v;
  const tx=2*(y*vz-z*vy),ty=2*(z*vx-x*vz),tz=2*(x*vy-y*vx);
  return [vx+w*tx+y*tz-z*ty, vy+w*ty+z*tx-x*tz, vz+w*tz+x*ty-y*tx];
}
const L=[0.32,-0.45,0.834];
function draw(){
  const f=D.frames[frame];
  const camx=document.getElementById('follow').checked?f.root_x:D.frames[0].root_x;
  ctx.clearRect(0,0,cv.width,cv.height);
  // floor grid every 0.1 m with x labels every 0.5 m
  ctx.lineWidth=1;
  for(let gx=Math.ceil((camx-1.4)/0.1)*0.1; gx<camx+1.4; gx+=0.1){
    const a=project([gx,-0.5,0],camx),b=project([gx,0.5,0],camx);
    const major=Math.abs(gx/0.5-Math.round(gx/0.5))<1e-6;
    ctx.strokeStyle=major?'#3a3a44':'#26262e';
    ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
    if(major){ctx.fillStyle='#6d6d75';ctx.font='20px system-ui';
      ctx.fillText(gx.toFixed(1)+' m',a[0]+4,a[1]+24);}
  }
  for(let gy=-0.5;gy<=0.5;gy+=0.1){
    const a=project([camx-1.4,gy,0],camx),b=project([camx+1.4,gy,0],camx);
    ctx.strokeStyle='#26262e';ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
  }
  const tris=[];
  for(const g of D.geometry){
    if(g.mesh==='foot_bottom_tpu'&&!g.collision)continue;
    const pose=f.poses[g.body],p0=pose[0],q=pose[1];
    for(const t of g.triangles){
      const ws=t.map(v=>{const r=rot(q,v);return [r[0]+p0[0],r[1]+p0[1],r[2]+p0[2]]});
      const pr=ws.map(p=>project(p,camx));
      const ux=ws[1][0]-ws[0][0],uy=ws[1][1]-ws[0][1],uz=ws[1][2]-ws[0][2];
      const vx=ws[2][0]-ws[0][0],vy=ws[2][1]-ws[0][1],vz=ws[2][2]-ws[0][2];
      const n=[uy*vz-uz*vy,uz*vx-ux*vz,ux*vy-uy*vx];
      const len=Math.hypot(n[0],n[1],n[2])||1;
      const lighting=.45+.55*Math.abs((n[0]*L[0]+n[1]*L[1]+n[2]*L[2])/len);
      const c=g.rgba.slice(0,3).map(v=>Math.round(Math.min(1,v*lighting)*255));
      tris.push({pr,depth:(pr[0][2]+pr[1][2]+pr[2][2])/3,c});
    }
  }
  tris.sort((a,b)=>b.depth-a.depth);   // painter: farthest along view dir first
  for(const t of tris){
    ctx.fillStyle=`rgb(${t.c[0]},${t.c[1]},${t.c[2]})`;
    ctx.beginPath();ctx.moveTo(t.pr[0][0],t.pr[0][1]);
    ctx.lineTo(t.pr[1][0],t.pr[1][1]);ctx.lineTo(t.pr[2][0],t.pr[2][1]);
    ctx.closePath();ctx.fill();
  }
  document.getElementById('cap').innerHTML=
   `t ${f.time_s.toFixed(2)} s · x ${f.root_x.toFixed(3)} m (target ${(D.command_mps*f.time_s).toFixed(2)}) · tilt ${f.tilt_deg.toFixed(1)}°`+
   `<span class="chip ${f.contacts[0]?'on':'off'}">L ${f.contacts[0]?'down':'up'}</span>`+
   `<span class="chip ${f.contacts[1]?'on':'off'}">R ${f.contacts[1]?'down':'up'}</span>`;
  scrub.value=frame;
}
function loop(now){
  const dt=(now-last)/1000;last=now;
  if(playing){
    acc+=dt*parseFloat(document.getElementById('speed').value);
    while(acc>=D.frame_dt_s){acc-=D.frame_dt_s;frame=(frame+1)%D.frames.length;}
  }
  draw();requestAnimationFrame(loop);
}
document.getElementById('play').onclick=e=>{playing=!playing;e.target.textContent=playing?'Pause':'Play'};
scrub.oninput=e=>{playing=false;document.getElementById('play').textContent='Play';frame=+e.target.value};
requestAnimationFrame(loop);
</script></body></html>
"""

if __name__ == "__main__":
    main()

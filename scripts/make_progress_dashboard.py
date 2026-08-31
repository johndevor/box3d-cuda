#!/usr/bin/env python3
"""Regenerate runs/progress.html from runs/flat-*/metrics.jsonl.

Usage: .venv/bin/python -B scripts/make_progress_dashboard.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [("flat-002", "baseline -> repaired solver"),
        ("flat-003", "phase-locked reward")]
EVENTS = [(591, "solver repair"), (1301, "phase-locked reward")]


def rnd(x, n=4):
    return None if x is None else round(x, n)


def load(name):
    train, evals = {}, []
    path = ROOT / "runs" / name / "metrics.jsonl"
    if path.exists():
        for line in path.open():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("kind") == "eval":
                skip = bool(d.get("skipped"))
                evals.append({"u": d["update"],
                              "ret": None if skip else rnd(d.get("eval_return_mean"), 1),
                              "len": None if skip else rnd(d.get("eval_len_mean"), 1)})
            elif d.get("kind") == "train":
                train[d["update"]] = d  # last occurrence wins (resume overlap)
    rows = [dict(u=u, rew=rnd(t.get("reward_mean")), eplen=rnd(t.get("ep_len_mean"), 1),
                 f=t.get("faults", 0), es=t.get("env_steps", 0))
            for u, t in sorted(train.items())]
    return rows, evals


def main():
    runs = []
    for name, label in RUNS:
        rows, evals = load(name)
        if rows:
            runs.append({"run": name, "label": label, "train": rows, "evals": evals})
    payload = json.dumps({"runs": runs, "events": EVENTS}, separators=(",", ":"))
    html = TEMPLATE.replace("__DATA__", payload)
    out = ROOT / "runs" / "progress.html"
    out.write_text(html)
    print(out)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>duck-grid-walk training progress</title>
<style>
.viz-root{
  color-scheme:light;
  --surface-1:#fcfcfb; --page:#f9f9f7;
  --ink-1:#0b0b0b; --ink-2:#52514e; --ink-3:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10);
  --s1:#2a78d6; --s2:#eb6834;
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])) .viz-root{
    color-scheme:dark;
    --surface-1:#1a1a19; --page:#0d0d0d;
    --ink-1:#ffffff; --ink-2:#c3c2b7; --ink-3:#898781;
    --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
    --s1:#3987e5; --s2:#d95926;
  }
}
:root[data-theme="dark"] .viz-root{
  color-scheme:dark;
  --surface-1:#1a1a19; --page:#0d0d0d;
  --ink-1:#ffffff; --ink-2:#c3c2b7; --ink-3:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
  --s1:#3987e5; --s2:#d95926;
}
body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.viz-root{background:var(--page);color:var(--ink-1);min-height:100vh;padding:28px 32px 48px}
h1{font-size:19px;margin:0 0 2px;font-weight:650}
.sub{color:var(--ink-2);font-size:13px;margin:0 0 18px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:18px;max-width:980px}
.tile{background:var(--surface-1);border:1px solid var(--ring);border-radius:10px;padding:12px 14px}
.tile .k{font-size:11.5px;color:var(--ink-2);letter-spacing:.01em}
.tile .v{font-size:24px;font-weight:650;margin-top:2px}
.tile .d{font-size:11.5px;color:var(--ink-3);margin-top:2px}
.legend{display:flex;gap:18px;align-items:center;margin:0 0 10px;font-size:12.5px;color:var(--ink-2)}
.legend .chip{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.gridwrap{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:14px;max-width:980px}
.panel{background:var(--surface-1);border:1px solid var(--ring);border-radius:10px;padding:12px 12px 6px}
.panel h2{font-size:12.5px;font-weight:600;color:var(--ink-2);margin:0 0 4px}
svg{display:block;width:100%;height:auto}
.gl{stroke:var(--grid);stroke-width:1}
.ax{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--ink-3);font-size:10px}
.evt{stroke:var(--ink-3);stroke-width:1;stroke-dasharray:3 3;opacity:.7}
.evtlab{fill:var(--ink-3);font-size:9.5px}
.endlab{font-size:10.5px;font-weight:600;fill:var(--ink-2)}
.tt{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--ring);
    border-radius:8px;padding:7px 10px;font-size:11.5px;color:var(--ink-1);box-shadow:0 2px 10px rgba(0,0,0,.12);
    display:none;z-index:9;min-width:130px}
.tt .r{display:flex;justify-content:space-between;gap:12px}
.tt .lab{color:var(--ink-2)}
.tt .num{font-variant-numeric:tabular-nums}
details{max-width:980px;margin-top:16px;color:var(--ink-2);font-size:13px}
table{border-collapse:collapse;font-size:12px;margin-top:8px;background:var(--surface-1)}
th,td{border:1px solid var(--grid);padding:4px 10px;text-align:right;font-variant-numeric:tabular-nums;color:var(--ink-1)}
th{color:var(--ink-2);font-weight:600}
.toggle{float:right;font-size:12px;color:var(--ink-2);background:none;border:1px solid var(--ring);
        border-radius:6px;padding:4px 10px;cursor:pointer}
</style></head>
<body><div class="viz-root">
<button class="toggle" onclick="const r=document.documentElement;r.dataset.theme=(r.dataset.theme==='dark'?'light':'dark')">theme</button>
<h1>Open Duck &mdash; flat-floor walking, PPO training progress</h1>
<p class="sub">192 envs &times; 12 workers on M5 Pro &middot; 0.02 s policy steps &middot; dashed lines mark the
solver repair (u591) and the phase-locked stance reward / run fork (u1301). flat-001 (25 updates, superseded) omitted.</p>
<div class="tiles" id="tiles"></div>
<div class="legend" id="legend"></div>
<div class="gridwrap" id="grid"></div>
<div class="tt" id="tt"></div>
<details><summary>Data table (evals + every 100th training update)</summary><div id="tbl"></div></details>
<script>
const DATA = __DATA__;
const COLORS = ["var(--s1)","var(--s2)"];
const W=560,H=232,ML=46,MR=64,MT=16,MB=24;
const runs = DATA.runs;
const allU = runs.flatMap(r=>r.train.map(t=>t.u));
const umax = Math.max(...allU), umin = 0;
const X = u => ML + (u-umin)/(umax-umin)*(W-ML-MR);

function niceTicks(lo,hi,n=4){
  if(hi===lo){hi=lo+1}
  const span=hi-lo, step0=span/n, mag=Math.pow(10,Math.floor(Math.log10(step0)));
  const step=[1,2,2.5,5,10].map(m=>m*mag).find(s=>span/s<=n+0.5)||mag*10;
  const t0=Math.ceil(lo/step)*step, out=[];
  for(let t=t0;t<=hi+1e-9;t+=step)out.push(+t.toFixed(6));
  return out;
}
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;")}

const PANELS=[
 {key:"rew",  title:"Train reward per policy step (rollout mean, per update)", get:r=>r.train.map(t=>[t.u,t.rew]), fmt:v=>v.toFixed(3)},
 {key:"eval", title:"Deterministic eval return (every 25 updates; × = skipped on solver fault)", get:r=>r.evals.filter(e=>e.ret!=null).map(e=>[e.u,e.ret]), fmt:v=>v.toFixed(0), marker:true,
  skips:r=>r.evals.filter(e=>e.ret==null).map(e=>e.u)},
 {key:"eplen",title:"Mean episode length, policy steps (400 = full 8 s, no fall)", get:r=>r.train.map(t=>[t.u,t.eplen]), fmt:v=>v.toFixed(0), ymax:400},
 {key:"f",    title:"Solver faults per update (poisoned rollout shards)", get:r=>r.train.map(t=>[t.u,t.f]), fmt:v=>v.toFixed(0)},
];

const grid=document.getElementById("grid"), tt=document.getElementById("tt");
PANELS.forEach((p,pi)=>{
  const series=runs.map(r=>p.get(r));
  const ys=series.flat().map(d=>d[1]);
  const ylo=Math.min(0,...ys), yhiRaw=Math.max(...ys);
  const yhi=p.ymax?Math.max(p.ymax,yhiRaw):yhiRaw*1.06;
  const Y=v=>MT+(1-(v-ylo)/(yhi-ylo))*(H-MT-MB);
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(p.title)}">`;
  const yt=niceTicks(ylo,yhi);
  yt.forEach(t=>{s+=`<line class="gl" x1="${ML}" x2="${W-MR}" y1="${Y(t)}" y2="${Y(t)}"/>`+
    `<text class="tick" x="${ML-6}" y="${Y(t)+3}" text-anchor="end">${t>=1000?(t/1000)+"k":t}</text>`});
  niceTicks(umin,umax,5).forEach(t=>{s+=`<text class="tick" x="${X(t)}" y="${H-8}" text-anchor="middle">${t}</text>`});
  s+=`<line class="ax" x1="${ML}" x2="${W-MR}" y1="${Y(ylo)}" y2="${Y(ylo)}"/>`;
  DATA.events.forEach(([u,lab])=>{ if(u<=umax){
    s+=`<line class="evt" x1="${X(u)}" x2="${X(u)}" y1="${MT}" y2="${H-MB}"/>`;
    if(pi===0)s+=`<text class="evtlab" x="${X(u)+4}" y="${MT+9}">${esc(lab)}</text>`;}});
  series.forEach((pts,si)=>{
    if(!pts.length)return;
    if(p.marker){
      pts.forEach(([u,v])=>{s+=`<circle cx="${X(u)}" cy="${Y(v)}" r="3.5" fill="${COLORS[si]}" stroke="var(--surface-1)" stroke-width="2"/>`});
      s+=`<polyline fill="none" stroke="${COLORS[si]}" stroke-width="1.4" opacity=".5" points="${pts.map(([u,v])=>X(u)+","+Y(v)).join(" ")}"/>`;
    }else{
      s+=`<polyline fill="none" stroke="${COLORS[si]}" stroke-width="2" stroke-linejoin="round" points="${pts.map(([u,v])=>X(u)+","+Y(v)).join(" ")}"/>`;
    }
    const last=pts[pts.length-1];
    s+=`<text class="endlab" x="${X(last[0])+6}" y="${Y(last[1])+3}" fill="${COLORS[si]}">${p.fmt(last[1])}</text>`;
  });
  if(p.skips){runs.forEach((r,si)=>{p.skips(r).forEach(u=>{
    s+=`<text x="${X(u)}" y="${H-MB-2}" text-anchor="middle" font-size="9" fill="var(--ink-3)">×</text>`;})})}
  s+=`<line id="ch${pi}" x1="0" x2="0" y1="${MT}" y2="${H-MB}" stroke="var(--ink-3)" stroke-width="1" opacity="0" />`;
  s+=`</svg>`;
  const div=document.createElement("div");div.className="panel";
  div.innerHTML=`<h2>${esc(p.title)}</h2>`+s; grid.appendChild(div);
  const svg=div.querySelector("svg"), ch=div.querySelector(`#ch${pi}`);
  svg.addEventListener("mousemove",ev=>{
    const box=svg.getBoundingClientRect(), fx=(ev.clientX-box.left)/box.width*W;
    if(fx<ML||fx>W-MR){tt.style.display="none";ch.setAttribute("opacity",0);return}
    const u=Math.round(umin+(fx-ML)/(W-ML-MR)*(umax-umin));
    ch.setAttribute("x1",X(u));ch.setAttribute("x2",X(u));ch.setAttribute("opacity",.6);
    let rows="";
    runs.forEach((r,si)=>{
      const pts=p.get(r); if(!pts.length)return;
      let best=null,bd=1e18; for(const d of pts){const dd=Math.abs(d[0]-u); if(dd<bd){bd=dd;best=d}}
      if(best&&bd<= (p.marker?13:6))
        rows+=`<div class="r"><span class="lab"><span class="chip" style="background:${COLORS[si]};display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:5px"></span>${r.run}</span><span class="num">${p.fmt(best[1])}</span></div>`;
    });
    if(!rows){tt.style.display="none";return}
    tt.innerHTML=`<div class="r"><span class="lab">update</span><span class="num">${u}</span></div>`+rows;
    tt.style.display="block";
    tt.style.left=Math.min(ev.clientX+14,innerWidth-170)+"px"; tt.style.top=(ev.clientY+12)+"px";
  });
  svg.addEventListener("mouseleave",()=>{tt.style.display="none";ch.setAttribute("opacity",0)});
});

// legend
document.getElementById("legend").innerHTML=runs.map((r,i)=>
 `<span><span class="chip" style="background:${COLORS[i]}"></span>${r.run} &middot; ${esc(r.label)}</span>`).join("");

// tiles
const lastRun=runs[runs.length-1], lastT=lastRun.train[lastRun.train.length-1];
const goodEvals=runs.flatMap(r=>r.evals.filter(e=>e.ret!=null));
const lastEval=goodEvals[goodEvals.length-1]||{ret:NaN,u:"-"};
const bestEval=goodEvals.reduce((a,b)=>b.ret>a.ret?b:a,{ret:-1e9});
const tail=lastRun.train.slice(-100), tailF=tail.reduce((a,t)=>a+t.f,0);
const totalSteps=runs.reduce((a,r)=>a+(r.train.length?r.train[r.train.length-1].es:0),0);
document.getElementById("tiles").innerHTML=[
 ["Latest update", lastT.u.toLocaleString(), lastRun.run],
 ["Latest eval return", lastEval.ret.toFixed(1), "u"+lastEval.u+" · deterministic, 8 s"],
 ["Best eval return", bestEval.ret.toFixed(1), "u"+bestEval.u],
 ["Faults, last 100 updates", tailF, "was ~700 per 100 pre-repair"],
 ["Env steps simulated", (totalSteps/1e6).toFixed(1)+"M", "across both runs"],
].map(([k,v,d])=>`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");

// table view
let tb=`<table><tr><th>update</th><th>run</th><th>reward</th><th>ep len</th><th>faults</th><th>eval return</th></tr>`;
runs.forEach(r=>{
  const em=new Map(r.evals.filter(e=>e.ret!=null).map(e=>[e.u,e.ret]));
  r.train.filter(t=>t.u%100===0).forEach(t=>{
    tb+=`<tr><td>${t.u}</td><td>${r.run}</td><td>${t.rew}</td><td>${t.eplen}</td><td>${t.f}</td><td>${em.get(t.u)??""}</td></tr>`});
});
tb+="</table>";
document.getElementById("tbl").innerHTML=tb;
</script>
</div></body></html>
"""

if __name__ == "__main__":
    main()

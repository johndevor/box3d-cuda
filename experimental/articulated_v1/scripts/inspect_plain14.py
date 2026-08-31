#!/usr/bin/env python3
"""Run the native exact-duck adapter statically; no simulator, step or trajectory."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from articulated_v1 import duck,library,np,rot,Scene,fp,dp
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);a=p.parse_args()
f=duck();lib=library(a.library);results=[]
for row in f.record['frames']:
 qp=row['qpos'];q=np.r_[qp[:3],qp[4:7],qp[3],qp[7:]];v=np.array(row['qvel']);v[3:6]=rot(q[3:7])@v[3:6]
 rc,e=f.evaluate(lib,q,v)
 if rc:raise RuntimeError('native evaluate status '+str(rc))
 scene=Scene(lib,f,[q],[v])
 try:
  g=np.zeros(f.N,'f');g[-1]=1;dv=np.zeros(f.N,'f');bodydv=np.zeros((f.B,6),'d');effective=np.zeros(1,'d')
  if lib.av1_response(scene.h,0,fp(g),fp(dv),dp(bodydv),dp(effective)):raise RuntimeError('native response failed')
  results.append({'retained_checkpoint':row['frame'],'bodies':f.B,'hinges':f.J,'generalized_velocities':f.N,
   'total_mass_kg':sum(b.mass for b in f.body),'minimum_body_mass_eigenvalue':float(np.linalg.eigvalsh(e.mass)[0]),
   'maximum_abs_bias':float(max(abs(e.bias))),'root_response_to_last_hinge_unit_impulse':dv[:6].tolist(),
   'responding_massive_bodies':int(np.count_nonzero(np.linalg.norm(bodydv[1:],axis=1)>1e-9)),
   'steps':scene.capture().count.tolist()})
 finally:scene.close()
print(json.dumps({'schema':'cuda3.articulated-v1.static-demo/v1','new_duck_steps':0,'mujoco_queries':0,'results':results},indent=2))

"""One source-pinned, externally bounded CPU-native home-hold attempt.

Requires passed run_local gates with identical source/artifact hashes. The
parent records raw output and enforces60s; the worker records every internal
PRE/POST state. This script never runs MuJoCo or creates a GPU resource.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
LANE=ROOT/'experimental/integrated_duck_v1'
GATES={'joint_limit_violation_rad':.05,'penetration_m':.01,'max_joint_speed_rad_s':250.,'max_base_linear_speed_m_s':20.,'max_base_angular_speed_rad_s':250.}
REFERENCE=ROOT/'duck_model/reference/open-duck-zero-hold-cpu-v1'
REFERENCE_SHA='a6d578064b433e730612d7144742b706471e63a37e3c81bcbc24acb7a7203a58'
SOURCE_COMMIT='b9be205ac64488c23504ca42e5ec790337adeec3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    with Path(p).open('x') as f:json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def preflight(gate):
    data=json.loads(gate.read_text())
    if data.get('status')!='passed-local-native-cpu' or data.get('source_unchanged') is not True or len(data['commands'])!=14 or any(x['exit_code'] for x in data['commands']):raise ValueError('local gates not fully passed')
    for name,digest in data['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('post-gate source drift '+name)
    library=gate.parent/('libintegrated_duck.dylib' if sys.platform=='darwin' else 'libintegrated_duck.so')
    if sha(library)!=data['artifacts'][library.name]:raise ValueError('native library drift')
    if sha(REFERENCE/'cpu-result.json')!=REFERENCE_SHA:raise ValueError('reference drift')
    reference=json.loads((REFERENCE/'cpu-result.json').read_text())
    if reference['gates']!=GATES or reference['source_commit']!=SOURCE_COMMIT or len(reference['frames'])!=501:raise ValueError('reference contract')
    home=reference['frames'][0]['motor_targets']
    if any(f['effective_controls']!=home or f['motor_targets']!=home for f in reference['frames']):raise ValueError('delay no-op assumption invalid')
    if reference['delay_frames']!=[[0,1,2,0,1,2,0,1][i%8] for i in range(500)]:raise ValueError('delay cycle drift')
    return data,library,reference

def worker(gate,out):
    gate_data,library,reference=preflight(gate)
    sys.path.insert(0,str(LANE));import native
    import numpy as np
    lib=native.library(library);scene=None;start=time.monotonic()
    record={'schema':'box3d.integrated_duck.home_hold/1','status':'running','backend':'native coupled CPU',
      'cuda_execution':False,'provider_access':False,'policy_executed':False,'learned_gait':False,
      'standing_qualified':False,'new_mujoco_steps':0,'source_commit':SOURCE_COMMIT,'gates':GATES,
      'reference_sha256':REFERENCE_SHA,'local_gates_sha256':sha(gate),'library_sha256':sha(library),
      'requested_seconds':10,'wall_cap_seconds':60,'dt':.002,'action_repeat':10,'maximum_solver_iterations':4096,
      'impulse_tolerance':1e-8,'behavior_scope':'health only, not independently qualified standing/walking',
      'frames':[],'accepted_internal_steps':0,'attempted_internal_steps':0}
    def manifolds(values):
        return [{'count':m.count,'normal':list(m.normal),'tangent1':list(m.tangent1),'tangent2':list(m.tangent2),
          'points':[{'feature':x.feature,'point':list(x.point),'depth':x.depth,'normal_impulse':x.normal_impulse,'tangent_impulse':list(x.tangent_impulse)} for x in m.points[:m.count]]} for m in values]
    def state(x):
        return {'qpos':x.q[0].tolist(),'velocity':x.v[0].tolist(),'warm_force':x.warm[0].tolist(),
          'time_s':float(x.time[0]),'step_count':int(x.count[0]),'bodies':[list(b.state) for b in x.bodies],
          'pre_contact_cache':manifolds(x.cache),'current_geometry':manifolds(x.geometry)}
    def health(x):
        limits=np.array([[l.lower,l.upper] for l in scene.limits]);q=x.q[0];v=x.v[0]
        violations=np.maximum(np.maximum(limits[:,0]-q[7:],q[7:]-limits[:,1]),0)
        pen=max([0.]+[float(p.depth) for m in x.cache for p in m.points[:m.count]])
        postpen=max([0.]+[float(p.depth) for m in x.geometry for p in m.points[:m.count]])
        metrics={'joint_limit_violation_rad':float(np.max(violations)),'penetration_m':pen,
          'max_joint_speed_rad_s':float(np.max(np.abs(v[6:]))),'max_base_linear_speed_m_s':float(np.linalg.norm(v[:3])),
          'max_base_angular_speed_rad_s':float(np.linalg.norm(v[3:6]))}
        finite=all(np.isfinite(getattr(x,n)).all() for n in ['q','v','warm','time']) and all(math.isfinite(z) for b in x.bodies for z in b.state)
        up=1-2*(q[3]**2+q[4]**2)
        return {'finite':bool(finite),'metrics':metrics,'joint_limit_violation':violations.tolist(),
          'post_geometry_penetration_m':postpen,'tilt_rad':math.acos(min(1,max(-1,float(up)))),
          'height_m':float(q[2]),'horizontal_drift_m':float(np.linalg.norm(q[:2]))}
    def checkpoint(x,index,publish=True):
        h=health(x);q=x.q[0]
        frame={'frame':len(record['frames']),'internal_step':index,'time_s':float(x.time[0]),
          'base_pose':[float(z) for z in [*q[:3],q[6],*q[3:6]]],'joint_q':q[7:].tolist(),
          'joint_qdot':x.v[0,6:].tolist(),'principal_bodies':[list(b.state) for b in x.bodies],
          'contacts':[m.count>0 for m in x.cache[:2]],'contact_details':[{'pair':i,'position':list(p.point),
          'depth':p.depth,'normal_impulse':p.normal_impulse,'tangent_impulse':list(p.tangent_impulse)} for i,m in enumerate(x.cache) for p in m.points[:m.count]],**h}
        if publish:record['frames'].append(frame)
        return frame
    try:
        print('before exact native registration',flush=True);scene,model=native.duck_scene(lib);print('after exact native registration',flush=True)
        before=scene.read();home=np.array(reference['frames'][0]['motor_targets'],dtype='d')[None,:]
        expected=reference['frames'][0];bp=expected['base_pose'];expected_q=np.array([*bp[:3],*bp[4:],bp[3],*expected['joint_q']]);expected_v=np.array(expected['qvel'],dtype='d');expected_v[3:6]=native.av.rot(expected_q[3:7])@expected_v[3:6]
        if not np.array_equal(before.q[0],expected_q) or not np.array_equal(before.v[0],expected_v):raise ValueError('complete reset state mismatch')
        initial=checkpoint(before,0);write(out/'reset-checkpoint.json',initial)
        if not initial['finite'] or before.time[0]!=0 or before.count[0]!=0 or any(initial['metrics'][k]>v for k,v in GATES.items()):raise ValueError('reset health gate')
        with (out/'trace.jsonl').open('x',buffering=1) as trace:
            trace.write(json.dumps({'phase':'RESET','display':initial},separators=(',',':'),allow_nan=False)+'\n');trace.flush()
            for index in range(1,5001):
                control=(index-1)//10+1;record['attempted_internal_steps']=index
                pre=state(before);h=scene.f.hinge
                actuator=[float(np.clip(float(a.kp)*(float(np.clip(home[0,j],scene.limits[j].lower,scene.limits[j].upper))-before.q[0,7+j])-float(a.kv)*before.v[0,6+j],-float(a.cap),float(a.cap))) for j,a in enumerate(h)]
                passive=[-float(a.damping)*before.v[0,6+j] for j,a in enumerate(h)]
                begin={'phase':'BEGIN','internal_step':index,'control_frame':control,'delay_frames':reference['delay_frames'][control-1],
                  'action':[0.]*14,'effective_targets':home[0].tolist(),'actuator_force':actuator,'passive_force':passive,'pre':pre}
                trace.write(json.dumps(begin,separators=(',',':'),allow_nan=False)+'\n');trace.flush()
                rc,diagnostics=scene.step(dt=.002,target=home,max_iterations=4096,tolerance=1e-8)
                after=scene.read();post=state(after);health_now=health(after)
                row={'phase':'END','internal_step':index,'control_frame':control,'delay_frames':reference['delay_frames'][control-1],
                  'action':[0.]*14,'effective_targets':home[0].tolist(),'actuator_force':actuator,'passive_force':passive,
                  'pre':pre,'post':post,'status':rc,'diagnostics':diagnostics,'health':health_now,
                  'display':checkpoint(after,index if rc==0 else index-1,False)}
                trace.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                if rc:
                    record['rollback_exact']=after.bytes()==before.bytes();record['first_failure']={'internal_step':index,'status':rc,'diagnostics':diagnostics}
                    if record['frames'][-1]['internal_step']!=index-1:checkpoint(before,index-1)
                    raise RuntimeError('native step rejected: '+json.dumps(record['first_failure']))
                record['accepted_internal_steps']=index
                if not health_now['finite']:checkpoint(after,index);raise RuntimeError('nonfinite native state')
                if index%10==0:
                    hnow=checkpoint(after,index)
                    print('checkpoint '+str(control)+' time='+str(after.time[0])+' points='+str(diagnostics[0]['contact_points'])+' iterations='+str(diagnostics[0]['iterations']),flush=True)
                    if abs(after.time[0]-control*.02)>1e-10:raise RuntimeError('clock mismatch')
                    for key,bound in GATES.items():
                        if hnow['metrics'][key]>bound:raise RuntimeError('health gate '+key+': '+str(hnow['metrics'][key])+' > '+str(bound))
                before=after
        record['status']='passed-bounded-native-cpu-health-only'
    except Exception as e:
        record['status']='rejected';record['failure']=str(e);traceback.print_exc()
    finally:
        if scene:scene.close()
        record['wall_seconds']=time.monotonic()-start
        record['simulated_seconds']=record['accepted_internal_steps']*.002
        record['maxima']={key:max([0.]+[f['metrics'][key] for f in record['frames']]) for key in GATES}
        record['maximum_tilt_deg']=max([0.]+[f['tilt_rad']*180/math.pi for f in record['frames']])
        record['maximum_horizontal_drift_m']=max([0.]+[f['horizontal_drift_m'] for f in record['frames']])
        record['mujoco_imported']='mujoco' in sys.modules
        if record['mujoco_imported']:record['status']='rejected';record['failure']='unexpected simulator import'
        write(out/'result.json',record)
        print(json.dumps({k:record[k] for k in ['status','simulated_seconds','maximum_tilt_deg','maximum_horizontal_drift_m']}),flush=True)
    return 0 if record['status']=='passed-bounded-native-cpu-health-only' else 1

def recover_prefix(out,code,gate,lib):
    """A crash/timeout never creates a guessed POST. Reuse complete trace rows."""
    frames=[];last=None;accepted=0;attempted=0
    reset=out/'reset-checkpoint.json'
    if reset.exists():frames.append(json.loads(reset.read_text()))
    trace=out/'trace.jsonl'
    if trace.exists():
        with trace.open() as f:
            for line in f:
                try:r=json.loads(line)
                except json.JSONDecodeError:break # preserve raw truncated suffix
                if r.get('phase')=='BEGIN':attempted=max(attempted,r['internal_step'])
                if r.get('phase')=='END' and r['status']==0:
                    accepted=r['internal_step'];last=r['display']
                    if accepted%10==0:frames.append(last)
    if last is not None and (not frames or frames[-1]['internal_step']!=last['internal_step']):frames.append(last)
    for i,f in enumerate(frames):f['frame']=i
    record={'schema':'box3d.integrated_duck.home_hold/1','status':'rejected','failure':'worker terminated, exit='+str(code)+'; complete trace prefix only',
      'backend':'native coupled CPU','cuda_execution':False,'provider_access':False,'policy_executed':False,'learned_gait':False,
      'standing_qualified':False,'new_mujoco_steps':0,'source_commit':SOURCE_COMMIT,'gates':GATES,'reference_sha256':REFERENCE_SHA,
      'local_gates_sha256':sha(gate),'library_sha256':sha(lib),'frames':frames,'accepted_internal_steps':accepted,'attempted_internal_steps':attempted,
      'simulated_seconds':accepted*.002,'recovered_prefix_only':True,'fabricated_post_state':False,
      'maxima':{k:max([0.]+[f['metrics'][k] for f in frames]) for k in GATES},
      'maximum_tilt_deg':max([0.]+[f['tilt_rad']*180/math.pi for f in frames]),
      'maximum_horizontal_drift_m':max([0.]+[f['horizontal_drift_m'] for f in frames])}
    write(out/'result.json',record)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gates',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if not a.gates.is_absolute() or not a.output.is_absolute():raise SystemExit('absolute evidence paths required')
    if a.worker:return worker(a.gates,a.output)
    gates,lib,_=preflight(a.gates)
    if a.output.exists():raise SystemExit('refuse existing attempt path')
    a.output.mkdir(parents=True,exist_ok=False)
    for name in gates['source_sha256']:
        target=a.output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    shutil.copyfile(a.gates,a.output/'local-gates.json');shutil.copyfile(lib,a.output/lib.name)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--gates',str(a.gates),'--output',str(a.output)]
    start=time.monotonic()
    with (a.output/'stdout.log').open('xb') as stdout,(a.output/'stderr.log').open('xb') as stderr:
        try:code=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=60,check=False).returncode
        except subprocess.TimeoutExpired:code=124
    if not (a.output/'result.json').exists():recover_prefix(a.output,code,a.gates,lib)
    unchanged=all(sha(ROOT/name)==digest for name,digest in gates['source_sha256'].items()) and sha(lib)==gates['artifacts'][lib.name]
    evidence={'schema':'box3d.integrated_duck.one_attempt/1','command':command,'exit_code':code,'wall_seconds':time.monotonic()-start,
      'source_unchanged':unchanged,'retry':False,'cuda_execution':False,'provider_access':False,'maximum_wall_seconds':60,
      'base_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'source_sha256':gates['source_sha256'],'artifacts':{x.name:sha(x) for x in a.output.iterdir() if x.is_file()}}
    evidence['status']='passed-health-only' if code==0 and unchanged else 'rejected';write(a.output/'attempt.json',evidence)
    print(json.dumps({'status':evidence['status'],'path':str(a.output/'attempt.json'),'sha256':sha(a.output/'attempt.json')}))
    return code if code else int(not unchanged)
if __name__=='__main__':raise SystemExit(main())

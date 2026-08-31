"""Explicitly authorized THREE static retained-state queries; NEVER a trajectory."""
import time
START=time.monotonic()
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import mujoco
P=argparse.ArgumentParser();P.add_argument('--reference',type=Path,required=True);P.add_argument('--output',type=Path,required=True);a=P.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound():
 if time.monotonic()-START>=55:raise RuntimeError('static query inner wall limit')
assert mujoco.__version__=='3.3.7' and np.__version__=='2.4.6'
a.output.mkdir(parents=True,exist_ok=False)
r=a.reference/'open-duck-zero-hold-cpu-v1';g=a.reference/'open-duck-native-compat-v1/geometry-goldens.json'
assert sha(r/'cpu-result.json')=='a6d578064b433e730612d7144742b706471e63a37e3c81bcbc24acb7a7203a58'
assert sha(r/'setup.json')=='e9c56c273355c30352be3f3b23ed224e7ce55acc0634af7dfe6a46ae522f0484'
assert sha(g)=='e52ba7d0f79434499d8fb6c2d611eb46ee12e2f32cb36258b38cd22959d0b08b'
setup=json.loads((r/'setup.json').read_text());record=json.loads((r/'cpu-result.json').read_text());mapping=json.loads(g.read_text())['mapping']
pins={}
for name,m in setup['asset_files'].items():
 p=r/'model'/name;assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256'];pins[name]=m
bound()
model=mujoco.MjModel.from_xml_path(str(r/'model/scene_flat_terrain.xml'));model.opt.timestep=.002
assert (model.nq,model.nv,model.nu)==(21,20,14)
assert np.array_equal(model.opt.gravity,[0,0,-9.81])
frames=[]
for index in [0,250,500]:
 bound();f=record['frames'][index];assert f['frame']==index
 data=mujoco.MjData(model);data.qpos[:]=f['qpos'];data.qvel[:]=f['qvel'];data.time=f['time_s']
 before=(data.qpos.copy(),data.qvel.copy(),float(data.time))
 # Only kinematics, COM transforms, inertia, velocity and inverse-bias subcomponents.
 # No forward solver, integration, actuator, contact or constraint solve.
 mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data);mujoco.mj_makeM(model,data)
 mujoco.mj_comVel(model,data);bias=np.empty(20);mujoco.mj_rne(model,data,0,bias)
 mass=np.empty((20,20));mujoco.mj_fullM(model,mass,data.qM)
 assert np.linalg.eigvalsh(mass)[0]>0 and np.max(np.abs(mass-mass.T))<1e-12
 bodies=[]
 for b in mapping['bodies'][1:]:
  bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,b['name']);assert bid>0
  jv=np.empty((3,20));jw=np.empty((3,20));mujoco.mj_jacBodyCom(model,data,jv,jw,bid)
  vel=np.empty(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,bid,vel,0)
  bodies.append({'id':b['id'],'name':b['name'],'position':data.xipos[bid].tolist(),'rotation':data.ximat[bid].reshape(3,3).tolist(),
                 'jacobian_local_root_angular':np.concatenate([jv,jw]).tolist(),'velocity_angular_then_linear':vel.tolist()})
 assert np.array_equal(before[0],data.qpos) and np.array_equal(before[1],data.qvel) and before[2]==data.time
 frames.append({'frame':index,'qpos':f['qpos'],'qvel':f['qvel'],'time':f['time_s'],
                'mass_including_armature_local_root_angular':mass.tolist(),'bias_local_root_angular':bias.tolist(),'bodies':bodies})
 bound()
package=Path(mujoco.__file__).parent
runtime={p.name:sha(p) for p in sorted(package.iterdir()) if p.is_file() and (p.suffix in ['.so','.dylib'] or p.name=='__init__.py')}
result={'schema':'cuda3.articulated-v1.static-reference/v1','frames':frames,'indices':[0,250,500],
        'mj_step_calls':0,'new_trajectory':False,'source_commit':record['source_commit'],'model_asset_pins':pins,
        'record_sha256':sha(r/'cpu-result.json'),'setup_sha256':sha(r/'setup.json'),'geometry_goldens_sha256':sha(g),
        'script_sha256':sha(Path(__file__)),'python':sys.version,'executable':sys.executable,
        'mujoco_version':mujoco.__version__,'numpy_version':np.__version__,'mujoco_runtime_sha256':runtime,
        'dof_armature':model.dof_armature.tolist(),'dof_damping':model.dof_damping.tolist(),
        'dof_frictionloss':model.dof_frictionloss.tolist(),'reference_qpos':model.qpos0.tolist(),
        'gravity':model.opt.gravity.tolist(),'elapsed_seconds':time.monotonic()-START,
        'calls_per_checkpoint':['mj_kinematics','mj_comPos','mj_makeM','mj_comVel','mj_rne(flg_acc=0)','mj_fullM','mj_jacBodyCom*15','mj_objectVelocity*15']}
bound();(a.output/'static-reference.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'passed-three-static-queries','elapsed_seconds':time.monotonic()-START,'reference_sha256':sha(a.output/'static-reference.json')}))

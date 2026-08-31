"""Export only actual native recorded checkpoints with retained source CAD.

No dynamics imports or time advancement. Reuses sealed stdlib source FK and
checks it against recorded native principal-COM poses before showing a frame.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
CAD_SHA='6cbb12676d0dec4f83af64a7ab6911943bde135cb1a5a50fc83e49de5eeba84e'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--asset-root',type=Path,required=True,help='historical CAD/FK bundle root (scripts/ and evidence/)');a=p.parse_args()
    OLD=a.asset_root.resolve()
    CAD=OLD/'evidence/open-duck-zero-hold-view-v1/open-duck-zero-hold-view.json'
    if not a.result.is_absolute() or not a.output.is_absolute() or a.output.exists():raise SystemExit('absolute result and fresh output required')
    if sha(CAD)!=CAD_SHA:raise ValueError('sealed CAD replay drift')
    data=json.loads(a.result.read_text());base=json.loads(CAD.read_text())
    if data['schema']!='box3d.integrated_duck.home_hold/1' or data['cuda_execution'] is not False or data['new_mujoco_steps']!=0 or data['learned_gait'] is not False:raise ValueError('recording scope')
    helpers={'scripts/export_open_duck_recorded_view.py':'94df276acf6c763e900f01eb43a92d3f97172efb3d6be1b1fca357c1c6eef411',
      'open_duck_plain14_candidate.py':'784ab945e9b0519dd831cd615ef0e83b728ead81789ea881735e324e00105838'}
    for name,digest in helpers.items():
        if sha(OLD/name)!=digest:raise ValueError('read-only exporter helper drift '+name)
    sys.path.insert(0,str(OLD));from scripts import export_open_duck_recorded_view as cad
    fk_bodies,_,identities=cad.load_model(OLD/'evidence/open-duck-zero-hold-cpu-v1/model/open_duck_mini_v2.xml')
    if identities!=base['source_file_sha256']:raise ValueError('CAD source XML/STL identity mismatch')
    mapping=json.loads((ROOT/'experimental/articulated_v1/fixtures/geometry-goldens.json').read_text())['mapping']
    body_by_name={b['name']:b for b in mapping['bodies']}
    peak=0.;frames=[]
    for f in data['frames']:
        poses=cad.forward_kinematics(fk_bodies,f)
        for body,pose in zip(base['bodies'],poses):
            if body['name']=='base':continue
            m=body_by_name[body['name']];recorded=f['principal_bodies'][m['id']]
            principal=cad.compose(pose,(m['source_COM'],m['inertial_quaternion_wxyz']))
            error=max(abs(principal[0][k]-recorded[k]) for k in range(3))
            rq=[recorded[6],*recorded[3:6]]
            for axis in [(1,0,0),(0,1,0),(0,0,1)]:error=max(error,max(abs(x-y) for x,y in zip(cad.rotate(principal[1],axis),cad.rotate(rq,axis))))
            if not math.isfinite(error) or error>1e-6:raise ValueError('CAD/native frame discrepancy '+str(error))
            peak=max(peak,error)
        frames.append({**f,'poses':poses})
    payload={'schema':'box3d.native-duck-recorded-replay/1','record_sha256':sha(a.result),'reference_sha256':base['candidate_sha256'],
      'native_status':data['status'],'native_failure':data.get('failure'),'simulated_seconds':data['simulated_seconds'],
      'cuda_execution':False,'viewer_simulates_physics':False,'policy_executed':False,'frames':frames,
      'geometry':base['geometry'],'bodies':base['bodies'],'reference_frames':base['frames'],
      'native_cad_max_difference':peak,'source_commit':data['source_commit'],'asset_notice':base['asset_notice'],
      'maximum_tilt_deg':data['maximum_tilt_deg'],'maximum_drift_m':data['maximum_horizontal_drift_m'],
      'health_maxima':data['maxima'],'library_sha256':data['library_sha256']}
    a.output.mkdir(parents=True,exist_ok=False)
    target=a.output/'replay.json';target.write_text(json.dumps(payload,separators=(',',':'),allow_nan=False)+'\n')
    template=(ROOT/'experimental/integrated_duck_v1/replay.html').read_text()
    (a.output/'index.html').write_text(template.replace('{{DATA_SHA}}',sha(target)))
    for name in ['asset-NOTICE','asset-LICENSE','hardware-LICENSE']:shutil.copyfile(CAD.parent/name,a.output/name)
    manifest={'schema':'box3d.native-duck-recorded-replay-manifest/1','record_sha256':sha(a.result),
      'native_cad_max_difference':peak,'visualization_tolerance':1e-6,'new_physics_steps':0,
      'read_only_helper_sha256':helpers,'files':{x.name:sha(x) for x in a.output.iterdir() if x.is_file()}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'path':str(a.output),'checkpoints':len(frames),'native_status':data['status'],'cad_max_difference':peak}))
if __name__=='__main__':main()

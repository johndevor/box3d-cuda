#!/usr/bin/env python3
"""Build/tests only. Never launches static MuJoCo query script or duck dynamics."""
import argparse,hashlib,json,os,platform,shutil,subprocess,sys
from pathlib import Path
LANE=Path(__file__).resolve().parent;ROOT=LANE.parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 cxx=shutil.which('clang++');cc=shutil.which('clang')
 if not cxx or not cc:raise SystemExit('existing clang/clang++ required; no installation')
 commands=[]
 def run(name,cmd,env=None):
  with (out/(name+'.stdout')).open('wb') as o,(out/(name+'.stderr')).open('wb') as e:r=subprocess.run(cmd,cwd=ROOT,stdout=o,stderr=e,env=env,timeout=60)
  commands.append({'name':name,'command':cmd,'exit_code':r.returncode})
  if r.returncode:raise RuntimeError(name+' failed; see complete logs')
 suffix='.dylib' if platform.system()=='Darwin' else '.so';lib=out/('libarticulated_v2'+suffix)
 common=['-std=c++17','-Wall','-Wextra','-Werror','-I',str(ROOT/'include'),'-I',str(LANE/'include'),'-I',str(ROOT/'experimental/articulated_v1/include')]
 sources=[str(LANE/'src/articulated_v2.cpp'),str(ROOT/'experimental/articulated_v1/src/articulated_v1.cpp'),str(ROOT/'csrc/experimental_joint_v1.cpp')]
 try:
  run('compiler',[cxx,'--version'])
  run('library',[cxx,*common,'-O2','-fPIC','-shared',*sources,'-o',str(lib)])
  run('native-build',[cxx,*common,'-O2',str(LANE/'tests/native_smoke.cpp'),str(lib),'-o',str(out/'native')])
  run('native-run',[str(out/'native')])
  run('c11-build',[cc,'-std=c11','-Wall','-Wextra','-Werror','-I',str(LANE/'include'),'-I',str(ROOT/'include'),'-I',str(ROOT/'proposals'),'-I',str(ROOT/'experimental/articulated_v1/include'),str(LANE/'tests/c_header.c'),'-o',str(out/'abi')])
  run('c11-run',[str(out/'abi')])
  run('sanitizer-build',[cxx,*common,'-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer',str(LANE/'tests/native_smoke.cpp'),*sources,'-o',str(out/'sanitizer')])
  run('sanitizer-run',[str(out/'sanitizer')])
  env=dict(os.environ,AV2_LIBRARY=str(lib),PYTHONDONTWRITEBYTECODE='1')
  run('python-tests',[sys.executable,'-B','-m','unittest','discover','-s',str(LANE/'tests'),'-p','test_v2.py','-v'],env)
  run('frozen-av1-tests',[sys.executable,'-B','-m','unittest','discover','-s',str(ROOT/'experimental/articulated_v1/tests'),'-p','test_articulated_v1.py','-v'],dict(os.environ,AV1_LIBRARY=str(lib),AV1_METRICS=str(out/'static-reference-metrics.json'),PYTHONDONTWRITEBYTECODE='1'))
  run('frozen-joint-tests',[sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_experimental_joint_v1.py','-v'],dict(os.environ,BOX3D_JOINT_V1_LIBRARY=str(lib),PYTHONDONTWRITEBYTECODE='1'))
 finally:
  result={'schema':'cuda3.articulated-v2.native-validation/v1','all_passed':len(commands)==11 and all(x['exit_code']==0 for x in commands),'commands':commands,'python':sys.version,'platform':platform.platform(),'mujoco_queries':0,'duck_dynamics_steps':0,'provider_calls':0,'cuda_execution_tested':False,'dependencies_installed':0,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}}
  (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'passed':result['all_passed'],'result':str(out/'result.json')}))
if __name__=='__main__':main()

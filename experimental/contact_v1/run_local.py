"""Exclusive file-backed local build/test run; no CUDA/provider/simulator.

Requires an existing clang++ and Python stdlib. --output must be absent.
Stops on the first failing command. Does not install or resolve dependencies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
LANE=ROOT/'experimental/contact_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve()
    if not args.output.is_absolute() or out.exists():raise SystemExit('new absolute output directory required')
    compiler=shutil.which('clang++');cc=shutil.which('clang')
    if not compiler or not cc:raise SystemExit('existing clang/clang++ required')
    out.mkdir(parents=True,exist_ok=False);lib=out/('libcontact_v1.dylib' if sys.platform=='darwin' else 'libcontact_v1.so')
    flags=['-std=c++17','-Wall','-Wextra','-Werror','-ffp-contract=off','-I',str(LANE/'include')]
    source=str(LANE/'src/contact_v1.cpp');cxx_test=str(LANE/'tests/contact_v1_tests.cpp')
    commands=[('toolchain',[compiler,'--version']),('build-shared',[compiler,*flags,'-fPIC','-shared',source,'-o',str(lib)]),
      ('build-host-tests',[compiler,*flags,source,cxx_test,'-o',str(out/'contact_tests')]),
      ('native-tests',[str(out/'contact_tests')]),('c11-header',[cc,'-std=c11','-Wall','-Wextra','-Werror','-I',str(LANE/'include'),str(LANE/'tests/c_header.c'),'-o',str(out/'c_header')]),
      ('c11-run',[str(out/'c_header')]),
      ('independent-response',[sys.executable,'-B',str(LANE/'tests/test_independent_response.py')]),
      ('model-translation',[sys.executable,'-B',str(LANE/'tests/test_model_translation.py')]),
      ('independent-geometry',[sys.executable,'-B',str(LANE/'tests/test_independent_geometry.py')]),
      ('build-sanitized',[compiler,*flags,'-fsanitize=address,undefined','-fno-omit-frame-pointer',source,cxx_test,'-o',str(out/'sanitized_tests')]),
      ('sanitized-tests',[str(out/'sanitized_tests')])]
    pins={str(f.relative_to(ROOT)):sha(f) for f in sorted(LANE.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
    result={'schema':'box3d.contact_v1.local_run/1','cuda_compile':False,'cuda_execution':False,'full_robot_steps':0,'provider_access':False,'python':sys.version,'platform':platform.platform(),'source_sha256':pins,'commands':[],'status':'running'}
    env=os.environ.copy();env['CONTACT_V1_LIBRARY']=str(lib);env['BOX3D_CONTACT_V1_LIBRARY']=str(lib)
    code=0
    for name,command in commands:
        start=time.monotonic()
        with (out/(name+'.stdout')).open('xb') as stdout,(out/(name+'.stderr')).open('xb') as stderr:
            try:r=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=60,check=False);code=r.returncode
            except subprocess.TimeoutExpired:code=124
        result['commands'].append({'name':name,'command':command,'exit_code':code,'elapsed_s':time.monotonic()-start,'stdout_sha256':sha(out/(name+'.stdout')),'stderr_sha256':sha(out/(name+'.stderr'))})
        if code:break
    after={str(f.relative_to(ROOT)):sha(f) for f in sorted(LANE.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
    if after!=pins:code=125
    result['source_unchanged']=after==pins;result['status']='passed-local-native-cpu' if code==0 else 'rejected';result['exit_code']=code
    result['artifacts']={f.name:sha(f) for f in out.iterdir() if f.is_file()}
    with (out/'result.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':result['status'],'jobs':len(result['commands']),'path':str(out/'result.json'),'result_sha256':sha(out/'result.json')}));return code
if __name__=='__main__':raise SystemExit(main())

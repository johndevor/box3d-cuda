"""Fresh, file-backed native CPU integration gates. No complete robot steps.

Uses installed clang and the current interpreter (existing NumPy required).
Stops first failure. No installation, CUDA/provider, or simulator trajectory.
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
LANE=ROOT/'experimental/integrated_duck_v1'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():
    roots=[ROOT/'experimental',ROOT/'csrc/experimental_joint_v1.cpp',ROOT/'csrc/experimental_joint_v1_shared.h',ROOT/'include/box3d_cuda/experimental_joint_v1.h']
    paths=[]
    for p in roots: paths.extend(p.rglob('*') if p.is_dir() else [p])
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths) if p.is_file() and '__pycache__' not in p.parts}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output
    if not out.is_absolute() or out.exists(): raise SystemExit('fresh absolute output directory required')
    compiler=shutil.which('clang++');cc=shutil.which('clang')
    if not compiler or not cc: raise SystemExit('existing clang toolchain required')
    import numpy
    out.mkdir(parents=True,exist_ok=False)
    lib=out/('libintegrated_duck.dylib' if sys.platform=='darwin' else 'libintegrated_duck.so')
    includes=[ROOT/x for x in ['experimental/integrated_duck_v1/include','experimental/contact_v1/include','experimental/articulated_v1/include','experimental/articulated_v2/include','include','csrc']]
    flags=['-std=c++17','-Wall','-Wextra','-Werror','-ffp-contract=off','-O2']
    for p in includes:flags.extend(['-I',str(p)])
    units=[ROOT/x for x in ['experimental/integrated_duck_v1/src/coupled_impulse_v1.cpp','experimental/integrated_duck_v1/src/integrated_duck_v1.cpp','experimental/contact_v1/src/contact_v1.cpp','experimental/articulated_v1/src/articulated_v1.cpp','experimental/articulated_v2/src/articulated_v2.cpp','csrc/experimental_joint_v1.cpp']]
    py=lambda f:[sys.executable,'-B',str(ROOT/f)]
    jobs=[('toolchain',[compiler,'--version']),('build-combined',[compiler,*flags,'-fPIC','-shared',*map(str,units),'-o',str(lib)]),
          ('header-c11',[cc,'-std=c11','-Wall','-Wextra','-Werror','-fsyntax-only','-x','c',*[x for p in includes for x in ['-I',str(p)]],str(LANE/'include/integrated_duck_v1.h')]),
          ('joint-regression',py('tests/test_experimental_joint_v1.py')),
          ('articulated-v1-regression',py('experimental/articulated_v1/tests/test_articulated_v1.py')),
          ('articulated-v2',py('experimental/articulated_v2/tests/test_v2.py')),
          ('contact-response',py('experimental/contact_v1/tests/test_independent_response.py')),
          ('contact-geometry',py('experimental/contact_v1/tests/test_independent_geometry.py')),
          ('contact-model-static',py('experimental/contact_v1/tests/test_model_translation.py')),
          ('contact-transactions',py('experimental/integrated_duck_v1/tests/test_contact_transactions.py')),
          ('coupled-impulse',py('experimental/integrated_duck_v1/tests/test_coupled_impulse.py')),
          ('integrated-synthetic',py('experimental/integrated_duck_v1/tests/test_integrated.py')),
          ('build-combined-sanitized',[compiler,*flags,'-fsanitize=address,undefined','-fno-omit-frame-pointer',*map(str,units),str(LANE/'tests/native_sanitized.cpp'),'-o',str(out/'integrated_sanitized')]),
          ('combined-sanitized',[str(out/'integrated_sanitized')])]
    before=sources();result={'schema':'box3d.integrated_duck.local-gates/1','source_sha256':before,
      'full_robot_steps':0,'gpu_execution':False,'provider_access':False,'simulator_execution':False,
      'python':sys.version,'numpy':numpy.__version__,'platform':platform.platform(),'commands':[]}
    env=os.environ.copy()
    for key in ['AV1_LIBRARY','AV2_LIBRARY','CONTACT_V1_LIBRARY','BOX3D_CONTACT_V1_LIBRARY','COUPLED_IMPULSE_LIBRARY','INTEGRATED_DUCK_LIBRARY','BOX3D_JOINT_V1_LIBRARY']:
        env[key]=str(lib)
    code=0
    for name,command in jobs:
        start=time.monotonic()
        with (out/(name+'.stdout')).open('xb') as stdout,(out/(name+'.stderr')).open('xb') as stderr:
            try: code=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=60,check=False).returncode
            except subprocess.TimeoutExpired: code=124
        result['commands'].append({'name':name,'command':command,'exit_code':code,'wall_seconds':time.monotonic()-start,'stdout_sha256':sha(out/(name+'.stdout')),'stderr_sha256':sha(out/(name+'.stderr'))})
        if code:break
    after=sources();result['source_unchanged']=before==after
    if before!=after:code=125
    result['status']='passed-local-native-cpu' if not code else 'rejected';result['exit_code']=code
    result['artifacts']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'jobs':len(result['commands']),'path':str(out/'result.json'),'sha256':sha(out/'result.json')}))
    return code
if __name__=='__main__':raise SystemExit(main())

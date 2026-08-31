"""Bounded, provider-free source-checkout verification. Never launches CUDA.

Requires existing Python 3.10+, numpy, pytest, clang; torch for training tests.
Evidence output must be a fresh absolute directory. Stop on the first failure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--without-training',action='store_true')
    args=parser.parse_args();out=args.output
    if not out.is_absolute() or out.exists():parser.error('fresh absolute --output required')
    out.mkdir(parents=True)
    py=[sys.executable,'-B']
    jobs=[('native',py+['experimental/integrated_duck_v1/run_local.py','--output',str(out/'native')],180),
          ('legacy',py+['-m','pytest','-q','tests'],120),
          ('contact-repair',py+['experimental/integrated_duck_v1/tests/test_nullnorm_repair.py'],90),
          ('saved-failures',py+['experimental/integrated_duck_v1/tests/test_saved_nonconvergence.py'],60),
          ('grid',py+['-m','unittest','discover','-s','experimental/duck_world_v1/tests','-v'],120),
          ('serial-cuda-source',py+['experimental/duck_cuda/tests/test_serial_parity.py'],180),
          ('provider-mocks',py+['-m','unittest','discover','-s','gpu/tests','-v'],60)]
    if args.without_training:
        for suite in ['env','eval']:
            jobs.append((suite,py+['-m','unittest','discover','-s',f'walk/{suite}/tests','-v'],120))
    else:jobs.append(('walk-and-trainer',py+['-m','unittest','discover','-s','walk','-v'],180))
    suffix='.dylib' if sys.platform=='darwin' else '.so'
    lib=str(out/'native'/('libintegrated_duck'+suffix))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    for key in ['AV1_LIBRARY','AV2_LIBRARY','CONTACT_V1_LIBRARY','BOX3D_CONTACT_V1_LIBRARY','COUPLED_IMPULSE_LIBRARY','INTEGRATED_DUCK_LIBRARY','BOX3D_JOINT_V1_LIBRARY']:env[key]=lib
    result={'schema':'box3d.duck.cpu-integration/1','cuda_execution':False,'provider_access':False,
            'python':sys.version,'platform':platform.platform(),'commands':[],'status':'running'}
    start=time.monotonic();code=0
    for name,command,bound in jobs:
        remaining=600-(time.monotonic()-start)
        if remaining<=0:code=124;break
        began=time.monotonic()
        with (out/(name+'.stdout')).open('xb') as stdout,(out/(name+'.stderr')).open('xb') as stderr:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            try:code=process.wait(timeout=min(bound,remaining))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();code=124
        entry={'name':name,'command':command,'exit_code':code,'seconds':time.monotonic()-began}
        for stream in ['stdout','stderr']:
            entry[stream+'_sha256']=hashlib.sha256((out/(name+'.'+stream)).read_bytes()).hexdigest()
        result['commands'].append(entry);print(json.dumps(entry),flush=True)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if code:break
    result.update(status='passed' if code==0 else 'rejected',exit_code=code,elapsed_seconds=time.monotonic()-start)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return code

if __name__=='__main__':raise SystemExit(main())

"""Fail-closed benchmark for persistent clipped OBB manifolds."""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
from factory_os.benchmarks import BenchmarkResult, CapabilitySet, write_result
from .extension import _validate_sat_pair_indices, load_extension, manifold_step
from .manifold_reference import (
    BENCHMARK_STEPS, CONTRACT_ID, DEFAULT_SEED, INITIAL_SLIDER_SPEED_MPS,
    MAX_FINAL_PENETRATION_M, MAX_MANIFOLD_POINTS, MAX_SLIDER_FINAL_SPEED_RATIO,
    MAX_SLIDER_SPEED_INCREASE_MPS, MAX_STACK_HEIGHT_ERROR_M,
    MAX_TAIL_ANGULAR_SPEED_RAD_S, MAX_TAIL_LINEAR_SPEED_MPS,
    MAX_TAIL_POSITION_JITTER_M, MIN_PERSISTENT_CONTACT_FRAMES,
    MIN_STACK_CENTER_GAP_M, PAIR_ROLES, TAIL_WINDOW_STEPS,
    SAT_AXIS_TIE_EPSILON_M,
    assert_valid_manifold_state, make_manifold_stack_state, step_manifold_reference,
)
from .sat_reference import SATConfig

PARITY_STEPS=30
STATE_TOLERANCE=1.5e-2
IMPULSE_TOLERANCE=2.0e-2
MIN_FINAL_CONTACT_POINTS_PER_PAIR=2

def arguments():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worlds',type=int,default=4096);p.add_argument('--steps',type=int,default=BENCHMARK_STEPS)
    p.add_argument('--warmup',type=int,default=8);p.add_argument('--seed',type=int,default=DEFAULT_SEED)
    return p.parse_args()

def _norm(values): return math.sqrt(sum(float(v)*float(v) for v in values))

def _new_trace():
    return {'stack_y':[],'stack_linear_speed':[],'stack_angular_speed':[],
            'pair_contacts':[],'slider_speed':[]}

def _record_trace(trace,state,contacts):
    world=state[0]
    trace['stack_y'].append([float(world[b][1]) for b in range(1,5)])
    trace['stack_linear_speed'].append(max(_norm(world[b][7:10]) for b in range(1,5)))
    trace['stack_angular_speed'].append(max(_norm(world[b][10:13]) for b in range(1,5)))
    trace['pair_contacts'].append([bool(value) for value in contacts[0]])
    trace['slider_speed'].append(abs(float(world[5][7])))

def _physical_gate(final,trace):
    state,_,penetration,ids,impulses,counts=final
    world=state[0];tail_y=trace['stack_y'][-TAIL_WINDOW_STEPS:]
    tail_linear=trace['stack_linear_speed'][-TAIL_WINDOW_STEPS:]
    tail_angular=trace['stack_angular_speed'][-TAIL_WINDOW_STEPS:]
    tail_contacts=trace['pair_contacts'][-TAIL_WINDOW_STEPS:]
    expected_y=(0.25,0.75,1.25,1.75)
    height_error=max(abs(float(world[b][1])-expected_y[b-1]) for b in range(1,5))
    min_gap=min(float(world[b+1][1])-float(world[b][1]) for b in range(1,4))
    position_jitter=max(max(row[b] for row in tail_y)-min(row[b] for row in tail_y) for b in range(4))
    persistent_frames=min(sum(row[p] for row in tail_contacts) for p in range(len(PAIR_ROLES)))
    slider=trace['slider_speed'];prior=[INITIAL_SLIDER_SPEED_MPS]+slider[:-1]
    slider_increase=max(max(0.0,current-before) for before,current in zip(prior,slider))
    slider_ratio=slider[-1]/INITIAL_SLIDER_SPEED_MPS
    max_pen=max(map(float,penetration[0]));final_counts=[int(value) for value in counts[0]]
    all_cached=all(any(int(v)!=0 for v in pair) for pair in ids[0])
    positive_normal=all(any(int(ids[0][p][s])!=0 and float(impulses[0][p][s][0])>0
                            for s in range(MAX_MANIFOLD_POINTS)) for p in range(len(PAIR_ROLES)))
    all_contacted=all(any(row[p] for row in trace['pair_contacts']) for p in range(len(PAIR_ROLES)))
    result={'all_pairs_contacted':all_contacted,'maximum_stack_height_error_m':height_error,
        'minimum_stack_center_gap_m':min_gap,'maximum_final_penetration_m':max_pen,
        'maximum_tail_position_jitter_m':position_jitter,
        'maximum_tail_linear_speed_mps':max(tail_linear),'maximum_tail_angular_speed_rad_s':max(tail_angular),
        'minimum_persistent_contact_frames':persistent_frames,'slider_final_speed_ratio':slider_ratio,
        'maximum_slider_speed_increase_mps':slider_increase,'final_contact_counts':final_counts,
        'all_pairs_have_multi_point_final_contact':all(value>=MIN_FINAL_CONTACT_POINTS_PER_PAIR for value in final_counts),
        'all_pairs_have_feature_cache':all_cached,'all_pairs_have_positive_normal_cache':positive_normal}
    result['passed']=(all_contacted and height_error<=MAX_STACK_HEIGHT_ERROR_M and
        min_gap>=MIN_STACK_CENTER_GAP_M and max_pen<=MAX_FINAL_PENETRATION_M and
        position_jitter<=MAX_TAIL_POSITION_JITTER_M and max(tail_linear)<=MAX_TAIL_LINEAR_SPEED_MPS and
        max(tail_angular)<=MAX_TAIL_ANGULAR_SPEED_RAD_S and
        persistent_frames>=MIN_PERSISTENT_CONTACT_FRAMES and slider_ratio<=MAX_SLIDER_FINAL_SPEED_RATIO and
        slider_increase<=MAX_SLIDER_SPEED_INCREASE_MPS and result['all_pairs_have_multi_point_final_contact'] and
        all_cached and positive_normal)
    return result

def _cpu_trace(seed,cold_start=False):
    state,mass,half,inertia,pairs,ids,imp=make_manifold_stack_state(1,seed=seed)
    trace=_new_trace();final=None;config=SATConfig()
    for _ in range(BENCHMARK_STEPS):
        state,contacts,pen,ids,imp,counts=step_manifold_reference(
            state,mass,half,inertia,pairs,ids,imp,config,steps=1,warm_start=not cold_start)
        _record_trace(trace,state,contacts);final=(state,contacts,pen,ids,imp,counts)
    return final,trace,(mass,half,pairs)

def run_cpu_correctness_gate(seed=DEFAULT_SEED):
    started=time.perf_counter();warm,trace,materials=_cpu_trace(seed);cold,cold_trace,_=_cpu_trace(seed,True)
    duration=time.perf_counter()-started
    assert_valid_manifold_state(warm[0],warm[2],max_penetration=MAX_FINAL_PENETRATION_M)
    warm_gate=_physical_gate(warm,trace);cold_gate=_physical_gate(cold,cold_trace)
    convergence=(warm_gate['maximum_tail_position_jitter_m']<=cold_gate['maximum_tail_position_jitter_m'] and
        warm_gate['maximum_tail_linear_speed_mps']<=cold_gate['maximum_tail_linear_speed_mps'] and
        warm_gate['maximum_tail_angular_speed_rad_s']<=cold_gate['maximum_tail_angular_speed_rad_s'] and
        (warm_gate['maximum_tail_position_jitter_m']<cold_gate['maximum_tail_position_jitter_m'] or
         warm_gate['maximum_tail_linear_speed_mps']<cold_gate['maximum_tail_linear_speed_mps'] or
         warm_gate['maximum_tail_angular_speed_rad_s']<cold_gate['maximum_tail_angular_speed_rad_s']))
    no_added_energy=warm_gate['maximum_slider_speed_increase_mps']<=MAX_SLIDER_SPEED_INCREASE_MPS
    gate={'passed':warm_gate['passed'] and convergence and no_added_energy,
        'cpu_gate_duration_seconds':duration,'cpu_warm_start_improves_tail_convergence':convergence,
        'cpu_warm_start_no_added_slider_energy':no_added_energy,
        **{f'cpu_{k}':v for k,v in warm_gate.items() if k!='passed'},
        **{f'cpu_cold_{k}':v for k,v in cold_gate.items() if k in (
            'maximum_tail_position_jitter_m','maximum_tail_linear_speed_mps','maximum_tail_angular_speed_rad_s')}}
    if not gate['passed']:
        raise RuntimeError('CPU manifold correctness failed; CUDA timing refused: '+json.dumps(gate,sort_keys=True))
    return gate,materials

def _ft(torch,x): return torch.tensor(x,dtype=torch.float32,device='cuda')
def _it(torch,x): return torch.tensor(x,dtype=torch.int64,device='cuda')

def _tensors(torch,worlds,seed):
    s,m,h,i,p,ids,imp=make_manifold_stack_state(worlds,seed=seed)
    return (_ft(torch,s),_ft(torch,m),_ft(torch,h),_ft(torch,i),_it(torch,p),_it(torch,ids),_ft(torch,imp),p)

def _run(torch,state,mass,half,inertia,pairs,ids,imp,config,steps):
    touched=torch.zeros((state.shape[0],pairs.shape[0]),dtype=torch.bool,device=state.device)
    pen=None;counts=None
    for _ in range(steps):
        state,c,pen,ids,imp,counts=manifold_step(state,mass,half,inertia,pairs,ids,imp,config)
        touched|=c.bool()
    return state,touched,pen,ids,imp,counts

def _cuda_final(torch,result):
    state,contacts,pen,ids,imp,counts=result
    data=(state.cpu().tolist(),contacts.cpu().tolist(),pen.cpu().tolist(),ids.cpu().tolist(),imp.cpu().tolist(),counts.cpu().tolist())
    # Preserve finite/quaternion/nonnegative validation here; the explicit
    # physical gates below own the metric threshold and its diagnostics.
    assert_valid_manifold_state(data[0],data[2],max_penetration=float('inf'))
    return data

def _timed_final_gate(data):
    penetration=data[2];counts=data[5]
    minimum_per_pair=[min(int(world[pair]) for world in counts) for pair in range(len(PAIR_ROLES))]
    failing_worlds=[
        index for index,world in enumerate(counts)
        if any(int(value)<MIN_FINAL_CONTACT_POINTS_PER_PAIR for value in world)
    ]
    max_penetration=max(max(map(float,world)) for world in penetration)
    multi_point=not failing_worlds
    penetration_ok=max_penetration<=MAX_FINAL_PENETRATION_M
    return {'passed':multi_point and penetration_ok,
        'all_pairs_have_multi_point_final_contact':multi_point,
        'minimum_final_contact_counts_per_pair':minimum_per_pair,
        'final_multi_point_failure_world_count':len(failing_worlds),
        'first_final_multi_point_failure_world_indices':failing_worlds[:20],
        'maximum_final_penetration_m':max_penetration,
        'final_penetration_within_threshold':penetration_ok}

def _cuda_trace_gate(torch,tensors,config):
    state,mass,half,inertia,pairs,ids,imp=tensors[:7];trace=_new_trace();final=None
    for _ in range(BENCHMARK_STEPS):
        state,contacts,pen,ids,imp,counts=manifold_step(state,mass,half,inertia,pairs,ids,imp,config)
        torch.cuda.synchronize()
        state_data=state.cpu().tolist();contact_data=contacts.cpu().tolist()
        _record_trace(trace,state_data,contact_data)
        final=(state,contacts,pen,ids,imp,counts)
    data=_cuda_final(torch,final)
    return _physical_gate(data,trace)

def main():
    a=arguments()
    if a.worlds<=0 or a.steps!=BENCHMARK_STEPS or a.warmup<0 or a.seed<0:
        raise ValueError('manifold benchmark requires positive worlds, nonnegative seed/warmup, and exactly {} steps'.format(BENCHMARK_STEPS))
    cpu_gate,materials=run_cpu_correctness_gate(a.seed)
    try: import torch
    except ImportError as exc: raise RuntimeError('CPU manifold correctness passed, but CUDA timing requires CUDA-enabled PyTorch') from exc
    if not torch.cuda.is_available(): raise RuntimeError('CPU manifold correctness passed, but CUDA timing requires a visible CUDA device')
    load_extension();config=SATConfig()
    # Exact short-prefix CPU/CUDA parity, including topology and cache state.
    src,m,h,i,p,cids,cimp=make_manifold_stack_state(1,seed=a.seed)
    expected=step_manifold_reference(src,m,h,i,p,cids,cimp,config,steps=PARITY_STEPS)
    tensors=_tensors(torch,1,a.seed);actual=_run(torch,*tensors[:7],config,PARITY_STEPS);torch.cuda.synchronize()
    es,ec,ep,eids,eimp,ecount=expected;ast,ac,ap,aids,aimp,acount=actual
    state_error=float((ast-_ft(torch,es)).abs().max().item());pen_error=float((ap-_ft(torch,ep)).abs().max().item())
    impulse_error=float((aimp-_ft(torch,eimp)).abs().max().item())
    contacts_equal=ac.cpu().tolist()==ec;ids_equal=aids.cpu().tolist()==eids;counts_equal=acount.cpu().tolist()==ecount
    parity=(state_error<=STATE_TOLERANCE and pen_error<=MAX_FINAL_PENETRATION_M and impulse_error<=IMPULSE_TOLERANCE and contacts_equal and ids_equal and counts_equal)
    if not parity: raise RuntimeError('CPU/CUDA manifold parity failed; timing refused: '+json.dumps({'state_error':state_error,'penetration_error':pen_error,'impulse_error':impulse_error,'contacts':contacts_equal,'features':ids_equal,'counts':counts_equal},sort_keys=True))
    # Full untimed CUDA physical gate before timing.
    gate_tensors=_tensors(torch,1,a.seed);cuda_gate=_cuda_trace_gate(torch,gate_tensors,config)
    if not cuda_gate['passed']: raise RuntimeError('CUDA manifold physical gate failed; timing refused: '+json.dumps(cuda_gate,sort_keys=True))
    # Compile/warm on the one-world contract, never on timed state.
    warm=_tensors(torch,1,a.seed)
    if a.warmup: _run(torch,*warm[:7],config,a.warmup)
    timed=_tensors(torch,a.worlds,a.seed);_validate_sat_pair_indices(timed[4],int(timed[0].shape[1]));torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats();start=torch.cuda.Event(enable_timing=True);finish=torch.cuda.Event(enable_timing=True)
    start.record();result=_run(torch,*timed[:7],config,a.steps);finish.record();torch.cuda.synchronize();duration=start.elapsed_time(finish)/1000
    timed_data=_cuda_final(torch,result)
    timed_gate=_timed_final_gate(timed_data)
    if not timed_gate['passed']: raise RuntimeError('timed manifold final safety gate failed: '+json.dumps(timed_gate,sort_keys=True))
    mass_meta,half_meta,pair_meta=materials
    correctness={**cpu_gate,'passed':cpu_gate['passed'] and parity and cuda_gate['passed'] and timed_gate['passed'],
        'cpu_cuda_state_maximum_absolute_error':state_error,'cpu_cuda_penetration_maximum_absolute_error_m':pen_error,
        'cpu_cuda_cache_impulse_maximum_absolute_error':impulse_error,'cpu_cuda_contact_parity':contacts_equal,
        'cpu_cuda_feature_id_parity':ids_equal,'cpu_cuda_contact_count_parity':counts_equal,
        **{f'cuda_gate_{k}':v for k,v in cuda_gate.items() if k!='passed'},
        **{f'timed_{k}':v for k,v in timed_gate.items() if k!='passed'},
        'scenario_seed':a.seed,'control_hz':120,'physics_substeps':config.substeps,
        'pair_order':[list(x) for x in pair_meta],'pair_roles':list(PAIR_ROLES),
        'half_extents_m':[list(x) for x in half_meta[0]],'inverse_mass_per_body':list(mass_meta[0]),
        'friction':config.friction,'restitution':config.restitution,'angular_damping':config.angular_damping,
        'maximum_manifold_points':MAX_MANIFOLD_POINTS,'warm_start_cache':True,
        'sat_axis_tie_epsilon_m':SAT_AXIS_TIE_EPSILON_M,
        'correctness_gate_steps':BENCHMARK_STEPS,'tail_window_steps':TAIL_WINDOW_STEPS,
        'gate_thresholds':{'benchmark_steps':BENCHMARK_STEPS,'tail_window_steps':TAIL_WINDOW_STEPS,
            'maximum_stack_height_error_m':MAX_STACK_HEIGHT_ERROR_M,
            'minimum_stack_center_gap_m':MIN_STACK_CENTER_GAP_M,
            'maximum_final_penetration_m':MAX_FINAL_PENETRATION_M,
            'maximum_tail_position_jitter_m':MAX_TAIL_POSITION_JITTER_M,
            'maximum_tail_linear_speed_mps':MAX_TAIL_LINEAR_SPEED_MPS,
            'maximum_tail_angular_speed_rad_s':MAX_TAIL_ANGULAR_SPEED_RAD_S,
            'minimum_persistent_contact_frames':MIN_PERSISTENT_CONTACT_FRAMES,
            'maximum_slider_final_speed_ratio':MAX_SLIDER_FINAL_SPEED_RATIO,
            'maximum_slider_speed_increase_mps':MAX_SLIDER_SPEED_INCREASE_MPS,
            'minimum_final_contact_points_per_pair':MIN_FINAL_CONTACT_POINTS_PER_PAIR}}
    report=BenchmarkResult(backend='box3d_cuda_stage4',backend_version='upstream-30c67b5+factory-v4',
        workload='fixed floor, four-box persistent stack, and independent friction slider',contract_id=CONTRACT_ID,
        device=torch.cuda.get_device_name(),worlds=a.worlds,bodies_per_world=6,steps=a.steps,duration_seconds=duration,
        capabilities=CapabilitySet(rigid_body_integration=True,static_plane_contacts=False,dynamic_contacts=True,articulated_joints=False,continuous_collision=False,ray_queries=False,camera_rendering=False,robot_manipulation=False),
        correctness=correctness,peak_memory_bytes=int(torch.cuda.max_memory_allocated()),notes=(
            'Five explicit fixed pairs use deterministic face clipping with at most four contacts and stable topology feature IDs.',
            'Normal and two tangent impulses persist by feature ID and are warm-started before iterative velocity solving.',
            'This contract has no broad phase, CCD, joints, robot manipulation, or production stacking claim.',))
    write_result(a.output,report);print(json.dumps(report.to_dict(),sort_keys=True,allow_nan=False));return 0

if __name__=='__main__': raise SystemExit(main())

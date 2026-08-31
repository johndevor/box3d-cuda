# One native CPU home-hold attempt

This is a new **native coupled CPU** experiment, not a MuJoCo rerun, CUDA
execution, trained standing, walking, or numerical cross-engine equivalence.
It is allowed only after all composed local tests and independent review pass.

- Exact plain14 Open Duck source `b9be205ac64488c23504ca42e5ec790337adeec3`.
- Same accepted root Z `0.16788827542191784` m, all other reset values unchanged.
- B16 / J14 / P3, real 18-vertex foot convex hulls; fixed infinite floor.
- Same source mass, principal inertia, armature .027, damping .56, joint
  friction .068, kp13.37, kv0, actuator effort cap3.23, pair mu .6/.6/1.
- Source soft joint-limit defaults, no hard angle clipping.
- 500 zero-action control frames, dt .02, ten internal dt .002 steps per frame.
  Same repeated delay cycle `[0,1,2,0,1,2,0,1]`; constant zero actions leave
  every delayed effective target at the same authored home target.
- Native contact law: inelastic, bias .2*max(depth−2e-6,0)/dt capped1 m/s.
  Full generalized mass including armature; non-associated normal/Coulomb solve.
- Solver budget4096 sweeps; scalar impulse-correction / tangent scaled-KKT
  tolerance1e-8; AV2 momentum and joint tolerance1e-8. Not MuJoCo's solver.
- At most10 simulated seconds,60s external wall cap, one attempt, no retry.
  Numerical solver failure stops immediately, preserving the previous accepted
  state and first failed PRE attempt. No fallback, extra reset, or tuning sweep.
- Unchanged CONTROL-CHECKPOINT health gates: joint limit error .05rad,
  penetration .01m, joint speed250rad/s, root linear speed20m/s, root angular
  speed250rad/s; finite state/forces and clock tolerance1e-10s.
  Both PRE solver contact penetration and POST geometry are recorded separately;
  the frozen penetration gate applies to returned PRE solver contacts.
- Record every internal PRE/POST generalized state, principal body state,
  PRE manifold/cache impulses, current POST geometry, controls, actuator/passive
  forces, solver diagnostics, clocks and health. Full stdout/stderr retained.
- Report root tilt/height/drift descriptively; these health gates do **not**
  establish standing. Only the foot meshes collide; torso/limbs remain visual.
- Replay shows discrete recorded native poses and labels the backend. It never
  synthesizes motion or disguises a rejected trajectory as a success.

Old reference/evidence, r3/machine-v1, public main and CUDA kernels are untouched.
No GPU/provider, dependency installation, training, merge, or push in this gate.

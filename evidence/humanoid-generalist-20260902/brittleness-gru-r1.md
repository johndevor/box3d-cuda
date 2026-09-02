| config | pins | pass | s4242/c0.50 | s4242/c0.75 | s4242/c1.00 | s7/c0.50 | s7/c0.75 | s7/c1.00 |
|---|---|---|---|---|---|---|---|---|
| nominal | authored | 6/6 | PASS q12 8.0s 4.621m | PASS q18 8.0s 5.625m | PASS q25 8.0s 5.727m | PASS q12 8.0s 4.469m | PASS q18 8.0s 5.503m | PASS q25 8.0s 5.705m |
| gravity_9.81 | gravity_scale=0.491 | 4/6 | PASS q12 8.0s 3.25m | PASS q18 8.0s 5.07m | fail q25 8.0s 4.768m | PASS q12 8.0s 3.527m | PASS q18 8.0s 5.078m | fail q25 8.0s 4.646m |
| gravity_12 | gravity_scale=0.6 | 6/6 | PASS q12 8.0s 3.696m | PASS q18 8.0s 5.134m | PASS q25 8.0s 5.298m | PASS q12 8.0s 3.47m | PASS q18 8.0s 5.234m | PASS q25 8.0s 5.301m |
| gravity_15 | gravity_scale=0.75 | 6/6 | PASS q12 8.0s 3.86m | PASS q18 8.0s 5.316m | PASS q25 8.0s 5.727m | PASS q12 8.0s 3.959m | PASS q18 8.0s 5.328m | PASS q25 8.0s 5.739m |
| gravity_18 | gravity_scale=0.9 | 6/6 | PASS q12 8.0s 4.48m | PASS q18 8.0s 5.617m | PASS q25 8.0s 5.599m | PASS q12 8.0s 4.174m | PASS q18 8.0s 5.478m | PASS q25 8.0s 5.551m |
| mass_x0.85 | mass_scale=0.85 | 5/6 | PASS q12 8.0s 4.776m | PASS q18 8.0s 5.689m | PASS q25 8.0s 6.02m | fail q10 8.0s 4.445m | PASS q18 8.0s 5.286m | PASS q25 8.0s 5.804m |
| mass_x1.15 | mass_scale=1.15 | 4/6 | PASS q12 8.0s 4.595m | PASS q18 8.0s 5.178m | fail q0 4.94s 4.159m | PASS q12 8.0s 4.549m | PASS q18 8.0s 5.242m | fail q0 3.94s 3.096m |
| friction_x0.7 | friction_scale=0.7 | 6/6 | PASS q12 8.0s 4.671m | PASS q18 8.0s 5.521m | PASS q25 8.0s 5.505m | PASS q12 8.0s 4.564m | PASS q18 8.0s 5.449m | PASS q25 8.0s 5.738m |
| friction_x1.3 | friction_scale=1.3 | 6/6 | PASS q12 8.0s 4.684m | PASS q18 8.0s 5.671m | PASS q25 8.0s 5.718m | PASS q12 8.0s 4.479m | PASS q18 8.0s 5.52m | PASS q25 8.0s 5.826m |
| kp_x0.85 | kp_scale=0.85 | 4/6 | PASS q12 8.0s 4.486m | PASS q19 8.0s 5.603m | fail q0 2.3s 1.572m | PASS q12 8.0s 4.216m | PASS q18 8.0s 5.437m | fail q0 2.72s 1.868m |
| kp_x1.15 | kp_scale=1.15 | 5/6 | PASS q12 8.0s 4.397m | PASS q18 8.0s 5.82m | PASS q25 8.0s 5.803m | fail q11 8.0s 4.199m | PASS q17 8.0s 5.577m | PASS q25 8.0s 5.929m |
| latency_1 | latency_steps=1 | 6/6 | PASS q12 8.0s 4.161m | PASS q18 8.0s 5.25m | PASS q25 8.0s 5.344m | PASS q12 8.0s 4.358m | PASS q18 8.0s 5.173m | PASS q25 8.0s 5.375m |
| latency_2 | latency_steps=2 | 4/6 | fail q0 3.02s 0.122m | fail q17 8.0s 4.812m | PASS q24 8.0s 6.058m | PASS q12 8.0s 4.478m | PASS q18 8.0s 4.92m | PASS q24 8.0s 5.986m |
| earth_light_soft | gravity_scale=0.491, mass_scale=0.85, kp_scale=0.85, friction_scale=0.7, latency_steps=1 | 6/6 | PASS q12 8.0s 3.201m | PASS q18 8.0s 4.796m | PASS q25 8.0s 5.417m | PASS q12 8.0s 3.17m | PASS q18 8.0s 4.965m | PASS q25 8.0s 5.41m |
| 2g_heavy_stiff_lag | mass_scale=1.15, kp_scale=1.15, friction_scale=1.3, latency_steps=2 | 2/6 | fail q0 4.28s 1.97m | fail q0 3.5s 1.895m | PASS q24 8.0s 5.701m | fail q0 5.14s 2.354m | fail q17 8.0s 4.859m | PASS q24 8.0s 5.785m |

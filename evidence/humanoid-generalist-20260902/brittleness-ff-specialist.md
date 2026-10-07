| config | pins | pass | s4242/c0.50 | s4242/c0.75 | s4242/c1.00 | s7/c0.50 | s7/c0.75 | s7/c1.00 |
|---|---|---|---|---|---|---|---|---|
| nominal | authored | 6/6 | PASS q12 8.0s 4.218m | PASS q18 8.0s 5.699m | PASS q25 8.0s 6.124m | PASS q12 8.0s 4.071m | PASS q18 8.0s 5.759m | PASS q25 8.0s 6.077m |
| gravity_9.81 | gravity_scale=0.491 | 0/6 | fail q9 8.0s 2.715m | fail q18 8.0s 3.32m | fail q18 8.0s 3.294m | fail q0 4.94s 0.706m | fail q14 8.0s 3.123m | fail q18 8.0s 2.967m |
| gravity_12 | gravity_scale=0.6 | 3/6 | PASS q12 8.0s 3.305m | PASS q18 8.0s 3.747m | fail q19 8.0s 3.553m | fail q12 8.0s 2.073m | PASS q17 8.0s 3.674m | fail q23 8.0s 3.929m |
| gravity_15 | gravity_scale=0.75 | 6/6 | PASS q12 8.0s 3.301m | PASS q18 8.0s 5.326m | PASS q25 8.0s 5.808m | PASS q12 8.0s 3.392m | PASS q18 8.0s 5.346m | PASS q25 8.0s 5.567m |
| gravity_18 | gravity_scale=0.9 | 6/6 | PASS q12 8.0s 4.079m | PASS q18 8.0s 5.565m | PASS q25 8.0s 6.065m | PASS q12 8.0s 3.53m | PASS q18 8.0s 5.619m | PASS q25 8.0s 6.156m |
| mass_x0.85 | mass_scale=0.85 | 6/6 | PASS q12 8.0s 3.841m | PASS q18 8.0s 5.244m | PASS q25 8.0s 6.063m | PASS q12 8.0s 3.835m | PASS q18 8.0s 5.246m | PASS q25 8.0s 6.003m |
| mass_x1.15 | mass_scale=1.15 | 5/6 | PASS q12 8.0s 4.666m | PASS q18 8.0s 5.446m | fail q0 3.2s 2.486m | PASS q12 8.0s 4.634m | PASS q18 8.0s 5.321m | PASS q25 8.0s 5.688m |
| friction_x0.7 | friction_scale=0.7 | 6/6 | PASS q12 8.0s 4.286m | PASS q18 8.0s 5.663m | PASS q25 8.0s 6.165m | PASS q12 8.0s 3.916m | PASS q18 8.0s 5.722m | PASS q25 8.0s 6.11m |
| friction_x1.3 | friction_scale=1.3 | 6/6 | PASS q12 8.0s 4.234m | PASS q18 8.0s 5.768m | PASS q25 8.0s 6.024m | PASS q12 8.0s 4.049m | PASS q18 8.0s 5.644m | PASS q25 8.0s 6.015m |
| kp_x0.85 | kp_scale=0.85 | 3/6 | PASS q12 8.0s 4.47m | PASS q19 8.0s 6.066m | fail q0 2.36s 1.594m | PASS q12 8.0s 4.446m | fail q0 7.54s 5.893m | fail q0 2.8s 1.973m |
| kp_x1.15 | kp_scale=1.15 | 6/6 | PASS q12 8.0s 4.365m | PASS q18 8.0s 5.709m | PASS q25 8.0s 5.909m | PASS q12 8.0s 3.885m | PASS q18 8.0s 5.814m | PASS q25 8.0s 6.103m |
| latency_1 | latency_steps=1 | 6/6 | PASS q12 8.0s 4.192m | PASS q18 8.0s 5.537m | PASS q25 8.0s 5.975m | PASS q12 8.0s 4.19m | PASS q18 8.0s 5.453m | PASS q25 8.0s 6.088m |
| latency_2 | latency_steps=2 | 2/6 | fail q0 6.1s 3.484m | fail q0 2.14s 1.719m | PASS q24 8.0s 6.865m | fail q0 6.74s 3.545m | fail q17 8.0s 5.49m | PASS q24 8.0s 6.523m |
| earth_light_soft | gravity_scale=0.491, mass_scale=0.85, kp_scale=0.85, friction_scale=0.7, latency_steps=1 | 1/6 | fail q11 8.0s 1.895m | fail q18 8.0s 3.341m | fail q23 8.0s 5.004m | fail q0 5.76s 0.499m | PASS q18 8.0s 3.716m | fail q25 8.0s 4.068m |
| 2g_heavy_stiff_lag | mass_scale=1.15, kp_scale=1.15, friction_scale=1.3, latency_steps=2 | 1/6 | fail q0 6.24s 3.683m | fail q0 2.14s 1.689m | fail q23 8.0s 6.564m | fail q0 3.22s 1.439m | fail q0 7.48s 5.141m | PASS q24 8.0s 6.301m |

| config | pins | pass | s4242/c0.50 | s4242/c0.75 | s4242/c1.00 | s7/c0.50 | s7/c0.75 | s7/c1.00 |
|---|---|---|---|---|---|---|---|---|
| nominal | authored | 6/6 | PASS q12 8.0s 4.088m | PASS q19 8.0s 5.224m | PASS q25 8.0s 5.88m | PASS q12 8.0s 3.992m | PASS q18 8.0s 5.176m | PASS q25 8.0s 5.847m |
| gravity_9.81 | gravity_scale=0.491 | 4/6 | PASS q12 8.0s 3.263m | PASS q18 8.0s 4.817m | fail q22 8.0s 4.678m | PASS q12 8.0s 3.154m | PASS q18 8.0s 4.712m | fail q25 8.0s 4.417m |
| gravity_12 | gravity_scale=0.6 | 6/6 | PASS q12 8.0s 3.485m | PASS q18 8.0s 5.155m | PASS q25 8.0s 5.181m | PASS q12 8.0s 3.454m | PASS q18 8.0s 5.039m | PASS q25 8.0s 4.936m |
| gravity_15 | gravity_scale=0.75 | 6/6 | PASS q12 8.0s 3.863m | PASS q18 8.0s 5.019m | PASS q25 8.0s 5.283m | PASS q12 8.0s 3.843m | PASS q18 8.0s 5.28m | PASS q25 8.0s 5.233m |
| gravity_18 | gravity_scale=0.9 | 6/6 | PASS q12 8.0s 3.991m | PASS q18 8.0s 5.222m | PASS q25 8.0s 5.619m | PASS q12 8.0s 3.894m | PASS q18 8.0s 5.275m | PASS q25 8.0s 5.619m |
| mass_x0.85 | mass_scale=0.85 | 6/6 | PASS q12 8.0s 4.264m | PASS q19 8.0s 5.496m | PASS q25 8.0s 6.189m | PASS q12 8.0s 4.204m | PASS q18 8.0s 5.373m | PASS q25 8.0s 6.223m |
| mass_x1.15 | mass_scale=1.15 | 4/6 | PASS q12 8.0s 3.817m | PASS q19 8.0s 4.848m | fail q0 3.3s 2.359m | PASS q12 8.0s 3.669m | PASS q18 8.0s 4.831m | fail q0 3.74s 2.95m |
| friction_x0.7 | friction_scale=0.7 | 6/6 | PASS q12 8.0s 3.916m | PASS q19 8.0s 5.191m | PASS q25 8.0s 5.79m | PASS q12 8.0s 4.043m | PASS q18 8.0s 5.138m | PASS q25 8.0s 5.745m |
| friction_x1.3 | friction_scale=1.3 | 5/6 | PASS q12 8.0s 4.02m | PASS q19 8.0s 5.244m | PASS q25 8.0s 5.826m | fail q11 8.0s 3.792m | PASS q18 8.0s 5.17m | PASS q25 8.0s 5.824m |
| kp_x0.85 | kp_scale=0.85 | 2/6 | PASS q12 8.0s 3.611m | PASS q19 8.0s 5.082m | fail q0 2.52s 1.77m | fail q11 8.0s 3.507m | fail q0 5.78s 4.168m | fail q0 2.6s 1.819m |
| kp_x1.15 | kp_scale=1.15 | 5/6 | fail q11 8.0s 4.128m | PASS q19 8.0s 5.321m | PASS q25 8.0s 5.693m | PASS q12 8.0s 3.824m | PASS q18 8.0s 5.281m | PASS q25 8.0s 5.696m |
| latency_1 | latency_steps=1 | 6/6 | PASS q12 8.0s 3.864m | PASS q18 8.0s 5.133m | PASS q25 8.0s 5.247m | PASS q12 8.0s 3.71m | PASS q18 8.0s 5.004m | PASS q25 8.0s 5.307m |
| latency_2 | latency_steps=2 | 5/6 | PASS q12 8.0s 3.899m | PASS q18 8.0s 4.844m | PASS q25 8.0s 5.684m | PASS q12 8.0s 3.956m | PASS q18 8.0s 4.911m | fail q24 8.0s 5.607m |
| earth_light_soft | gravity_scale=0.491, mass_scale=0.85, kp_scale=0.85, friction_scale=0.7, latency_steps=1 | 6/6 | PASS q12 8.0s 2.938m | PASS q18 8.0s 4.777m | PASS q24 8.0s 4.954m | PASS q12 8.0s 3.241m | PASS q18 8.0s 4.747m | PASS q25 8.0s 5.077m |
| 2g_heavy_stiff_lag | mass_scale=1.15, kp_scale=1.15, friction_scale=1.3, latency_steps=2 | 4/6 | fail q0 5.86s 2.6m | PASS q18 8.0s 4.856m | PASS q25 8.0s 5.585m | PASS q12 8.0s 4.346m | fail q17 8.0s 4.782m | PASS q25 8.0s 5.525m |

/* duck-grid-walk results dashboard — cards + pre-recorded 3D replay viewer.
   Works from file:// : data arrives as classic <script> files calling DGW.data(). */
(function () {
  'use strict';
  const M = DGW._d.manifest;
  const $ = (sel, root) => (root || document).querySelector(sel);
  const el = (tag, attrs, ...kids) => {
    const n = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') n.className = v;
      else if (k === 'html') n.innerHTML = v;
      else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined) n.setAttribute(k, v);
    }
    for (const k of kids.flat()) if (k !== null && k !== undefined) n.append(k.nodeType ? k : document.createTextNode(String(k)));
    return n;
  };
  const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const fmt = (x, d = 1) => (x == null || Number.isNaN(x)) ? '—' : Number(x).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: 0 });
  const FAMILY_COLOR = { duck: 'var(--accent-2)', humanoid: 'var(--accent)', generalist: 'var(--accent-5)', variants: 'var(--accent-4)', arm: 'var(--accent-3)' };
  const STATUS = {
    accepted: ['ACCEPTED', '✓'], candidate: ['CANDIDATE', '◔'], partial: ['PARTIAL', '◑'],
    milestone: ['MILESTONE', '◆'], failed: ['NOT ACCEPTED', '✕'],
  };

  // ------------------------------------------------------------ data loading
  function load(name, file) {
    return new Promise((resolve, reject) => {
      if (DGW._d[name]) return resolve(DGW._d[name]);
      (DGW._w[name] = DGW._w[name] || []).push(resolve);
      if (document.querySelector(`script[data-dgw="${name}"]`)) return;
      const s = document.createElement('script');
      s.src = file; s.dataset.dgw = name;
      s.onerror = () => reject(new Error('failed to load ' + file));
      document.head.appendChild(s);
    });
  }

  // ------------------------------------------------------------ page
  const app = $('#app');
  const wrap = el('div', { class: 'wrap' });
  app.append(wrap);
  const S = M.summary;

  wrap.append(el('header', { class: 'hdr' },
    el('div', null,
      el('h1', null, 'duck-grid-walk ', el('span', null, '· trained robots')),
      el('div', { class: 'sub' }, 'Every policy we have trained on the in-house physics stack, grouped by robot family. Click a card to open its 3D replays.'),
      el('div', { class: 'notice' }, el('span', { class: 'dot' }),
        'Replays are pre-recorded rollouts of the real actor in the real physics (deterministic serial lane, same episodes the frozen judge scores) — the viewer advances no physics.')),
    el('div', { class: 'built' }, `built ${esc(M.built_at)}`, el('br'), `${S.clips_total} clips · ${(S.clip_bytes / 1e6).toFixed(1)} MB recorded`)
  ));

  // summary strip
  const accepted = M.cards.filter(c => c.status === 'accepted');
  wrap.append(el('section', { class: 'strip' },
    el('div', { class: 'tile' }, el('div', { class: 'k' }, 'Robots onboarded'),
      el('div', { class: 'v' }, S.robots_onboarded, el('small', null, `robot bodies · ${M.families.length} families`)),
      el('div', { class: 'd' }, S.robots.join(' · '))),
    el('div', { class: 'tile' }, el('div', { class: 'k' }, 'Accepted policies'),
      el('div', { class: 'v' }, S.accepted_policies, el('small', null, `of ${S.policies_total} trained policies`)),
      el('div', { class: 'd' }, accepted.map(c => c.title.replace(/ — accepted$/, '')).join(' · '))),
    el('div', { class: 'tile' }, el('div', { class: 'k' }, 'GPU legs'),
      el('div', { class: 'v' }, S.gpu_training_legs, el('small', null, `trained · ${S.gpu_launches} sandbox launches`)),
      el('div', { class: 'd' }, `${fmt(S.gpu_training_hours, 1)} h accumulated training wall clock on ephemeral RTX 5090s`)),
    el('div', { class: 'tile' }, el('div', { class: 'k' }, 'Environment steps'),
      el('div', { class: 'v' }, fmt(S.env_steps_b, 1), el('small', null, 'billion')),
      el('div', { class: 'd' }, `${S.commits_total} commits · duck 08-31 → arm 09-02`))
  ));

  // onboarding cost table
  if (S.onboarding && S.onboarding.length) {
    const rows = S.onboarding;
    wrap.append(el('details', { class: 'cost', open: '' },
      el('summary', null, 'Onboarding cost per robot', el('span', { class: 'hint' }, 'first commit → result commit · from git log + runs/gpu metrics')),
      el('table', { class: 'tbl' },
        el('thead', null, el('tr', null,
          el('th', null, 'Robot'), el('th', null, 'First commit'), el('th', null, 'Result commit'),
          el('th', { class: 'num' }, 'Wall hours'), el('th', { class: 'num' }, 'Commits'),
          el('th', { class: 'num' }, 'GPU legs'), el('th', { class: 'num' }, 'GPU hours'),
          el('th', { class: 'num' }, 'Env-steps (B)'), el('th', null, 'Result'))),
        el('tbody', null, rows.map(r => el('tr', null,
          el('td', null, r.label),
          el('td', null, el('span', { class: 'mono' }, `${r.first_commit} ${r.first_date || ''}`)),
          el('td', null, el('span', { class: 'mono' }, `${r.result_commit} ${r.result_date || ''}`)),
          el('td', { class: 'num' }, fmt(r.wall_hours, 1)),
          el('td', { class: 'num' }, r.commits == null ? '—' : r.commits),
          el('td', { class: 'num' }, `${r.gpu_legs}`, el('span', { class: 'mono' }, ` /${r.gpu_launches}`)),
          el('td', { class: 'num' }, fmt(r.gpu_hours, 2)),
          el('td', { class: 'num' }, fmt(r.env_steps_b, 2)),
          el('td', null, r.result))))),
      el('div', { class: 'tblnote' }, 'Wall hours = clock time between the two commits (one person + agents, other work interleaved). Commits = subject-keyword matches in that window (approximate). GPU legs = sandbox legs that produced training metrics / total launches for that robot\'s specs; GPU hours = summed in-leg training wall clock (excludes sandbox build/boot).')
    ));
  }

  // families & cards
  const byFamily = {};
  for (const c of M.cards) (byFamily[c.family] = byFamily[c.family] || []).push(c);
  for (const f of M.families) {
    const cards = byFamily[f.id] || [];
    const sec = el('section', { class: 'family' },
      el('h2', null, el('span', { class: 'swatch', style: `background:${FAMILY_COLOR[f.id] || 'var(--accent)'}` }), f.title,
        el('span', { style: 'color:var(--text-muted);font-weight:500;font-size:13px' }, `${cards.length} ${cards.length === 1 ? 'policy' : 'policies'}`)),
      el('div', { class: 'blurb' }, f.blurb));
    const grid = el('div', { class: 'cards' });
    if (!cards.length) grid.append(el('div', { class: 'card', style: 'cursor:default;color:var(--text-muted)' }, 'No trained actor yet.'));
    for (const c of cards) grid.append(cardEl(c));
    sec.append(grid);
    wrap.append(sec);
  }
  if (M.problems && M.problems.length) {
    wrap.append(el('div', { class: 'problems' }, el('b', null, 'Build notes: '), M.problems.join(' · ')));
  }
  wrap.append(el('footer', null, 'Judges: walk/eval/gait.py, walk/eval/humanoid_gait.py, walk/eval/arm_reach_judge.py (frozen). Verdicts on cards are recomputed on the CPU serial lanes at build time by scripts/build_dashboard.py and cross-checked against committed acceptance JSONs. Duck CAD: Open Duck Mini v2 source meshes (licenses retained with the sealed evidence).'));

  function cardEl(c) {
    const st = STATUS[c.status] || STATUS.candidate;
    const v = c.verdict || {};
    const cells = (v.cells || []);
    const card = el('article', { class: 'card', tabindex: '0', role: 'button', 'aria-label': `${c.title}: open replays` },
      el('div', { class: 'top' },
        el('div', null, el('div', { class: 'robot' }, c.robot), el('h3', null, c.title), el('div', { class: 'variant' }, c.variant)),
        el('span', { class: `badge ${c.status}` }, el('span', { class: 'ic' }, st[1]), st[0])),
      el('div', { class: 'verdict' },
        el('span', { class: 'score' }, v.label || '—'),
        el('span', { class: 'of' }, c.verdict_sub || (c.family === 'arm' ? 'seeds × tiers, frozen reach judge' : 'seeds × commands, frozen gait judge'))),
      cells.length ? el('div', { class: 'cells', title: cells.map(x => `${x.cell}: ${x.passed ? 'PASS' : 'fail ' + (x.failed || []).join(',')}`).join('\n') },
        cells.map(x => el('span', { class: x.passed ? 'p' : 'f' }))) : null,
      c.verdict_note ? el('div', { class: 'lineage', style: 'color:var(--text-secondary)' }, c.verdict_note) : null,
      el('div', { class: 'metrics' },
        (c.metrics || []).map(([k, val]) => el('div', { class: 'm' }, el('div', { class: 'k' }, k), el('div', { class: 'v' }, val))),
        el('div', { class: 'm' }, el('div', { class: 'k' }, 'Clips'), el('div', { class: 'v' }, `${c.clips.length} recorded`)),
        el('div', { class: 'm' }, el('div', { class: 'k' }, 'Date'), el('div', { class: 'v' }, c.date))),
      el('div', { class: 'lineage', title: c.lineage }, c.lineage),
      el('div', { class: 'foot' }, el('span', { class: 'path', title: c.evidence }, c.evidence), el('span', { class: 'open' }, 'Open 3D replay ▸')));
    const open = () => openViewer(c.id);
    card.addEventListener('click', open);
    card.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
    return card;
  }

  // ------------------------------------------------------------ viewer
  const overlay = el('div', { class: 'overlay', onclick: e => { if (e.target === overlay) closeViewer(); } });
  const V = {};
  const viewer = el('div', { class: 'viewer', role: 'dialog', 'aria-modal': 'true' },
    el('div', { class: 'vh' },
      el('div', { class: 't' }, V.title = el('h3'), V.badge = el('span', { class: 'badge accepted' }),
        el('span', { class: 'rec' }, el('span', { class: 'dot' }), 'recorded replay · deterministic physics · no live simulation')),
      el('button', { class: 'close', onclick: closeViewer }, 'Close  esc')),
    V.stage = el('div', { class: 'stage' },
      V.hud = el('div', { class: 'hud' }),
      V.fell = el('div', { class: 'fell' }),
      V.legend = el('div', { class: 'legend' })),
    V.side = el('aside', { class: 'side' }),
    el('div', { class: 'controls' },
      V.play = el('button', { onclick: () => setPlaying(!P.playing) }, 'Pause'),
      V.speed = el('div', { class: 'speed' }, [0.5, 1, 2].map(s => el('button', { class: s === 1 ? 'active' : '', onclick: () => setSpeed(s), 'data-s': s }, `${s}×`))),
      V.scrub = el('input', { type: 'range', min: '0', max: '1', value: '0', oninput: e => { setPlaying(false); P.frame = +e.target.value; P.acc = 0; render(true); } }),
      V.time = el('span', { class: 'time' }, '0.00 s'),
      el('label', { class: 'follow' }, V.follow = el('input', { type: 'checkbox', checked: '' }), 'follow'))
  );
  overlay.append(viewer);
  document.body.append(overlay);
  document.addEventListener('keydown', e => {
    if (!overlay.classList.contains('open')) return;
    if (e.key === 'Escape') closeViewer();
    else if (e.key === ' ') { e.preventDefault(); setPlaying(!P.playing); }
    else if (e.key === 'ArrowRight') { setPlaying(false); step(+1); }
    else if (e.key === 'ArrowLeft') { setPlaying(false); step(-1); }
  });

  // three.js state
  const T = { renderer: null, scene: null, camera: null, controls: null, rigKey: null, rig: null, objs: null };
  const P = { card: null, clip: null, meta: null, frame: 0, playing: true, speed: 1, acc: 0, last: 0, lastRootX: null, raf: 0 };

  function ensureRenderer() {
    if (T.renderer) return;
    const r = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    r.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.outputEncoding = THREE.sRGBEncoding;
    V.stage.prepend(r.domElement);
    T.renderer = r;
    T.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 200);
    T.camera.up.set(0, 0, 1);
    T.controls = new THREE.OrbitControls(T.camera, r.domElement);
    T.controls.enableDamping = true; T.controls.dampingFactor = 0.12;
    T.controls.maxPolarAngle = Math.PI * 0.52;
    new ResizeObserver(resize).observe(V.stage);
  }
  function resize() {
    if (!T.renderer) return;
    const w = V.stage.clientWidth, h = V.stage.clientHeight;
    if (!w || !h) return;
    T.renderer.setSize(w, h, false);
    T.camera.aspect = w / h; T.camera.updateProjectionMatrix();
  }

  function mat(color, opts) { return new THREE.MeshStandardMaterial(Object.assign({ color, metalness: 0.15, roughness: 0.6 }, opts || {})); }

  function buildScene(rigKey, rig, clipMeta) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c0e11);
    scene.fog = null;
    const hemi = new THREE.HemisphereLight(0xdfe6ee, 0x2a3038, 0.85); scene.add(hemi);
    const sun = new THREE.DirectionalLight(0xffffff, 1.35);
    const scale = rig.kind === 'arm' ? 6 : (rigKey === 'duck' ? 0.8 : 3);
    sun.position.set(2.5 * scale, -1.5 * scale, 4 * scale);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera; sc.near = 0.1; sc.far = 40 * scale; sc.left = sc.bottom = -3 * scale; sc.right = sc.top = 3 * scale;
    sun.shadow.bias = -0.0005;
    scene.add(sun); scene.add(sun.target);
    // floor (z = 0, z-up world) + grid
    const gsz = rig.kind === 'arm' ? 16 : (rigKey === 'duck' ? 8 : 40);
    const gdiv = rig.kind === 'arm' ? 32 : (rigKey === 'duck' ? 80 : 80);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(gsz * 4, gsz * 4), new THREE.MeshStandardMaterial({ color: 0x14181d, roughness: 0.95, metalness: 0 }));
    floor.receiveShadow = true; floor.position.z = -0.0015; scene.add(floor);
    const grid = new THREE.GridHelper(gsz, gdiv, 0x3a4552, 0x232a33);
    grid.rotation.x = Math.PI / 2; grid.position.z = 0.0005; scene.add(grid);
    const axis = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0.002), rigKey === 'duck' ? 0.15 : 0.5, 0x5a6572, rigKey === 'duck' ? 0.03 : 0.1, rigKey === 'duck' ? 0.015 : 0.05);
    scene.add(axis);

    const objs = { bodies: [], feet: [], sun, scale };
    if (rig.kind === 'boxes') {
      rig.bodies.forEach((b, i) => {
        if (b.name === 'floor') { objs.bodies.push(null); return; }
        const isFoot = rig.feet.includes(i), isPelvis = i === rig.pelvis;
        const m = new THREE.Mesh(new THREE.BoxGeometry(b.half[0] * 2, b.half[1] * 2, b.half[2] * 2),
          mat(isFoot ? 0x3987e5 : isPelvis ? 0xd95926 : (b.name.includes('arm') ? 0x7f8b98 : 0x9aa5b1)));
        m.castShadow = true; m.receiveShadow = true;
        m.userData.base = m.material.color.getHex();
        scene.add(m); objs.bodies.push(m);
        if (isFoot) objs.feet.push(m);
      });
    } else if (rig.kind === 'mesh') {
      const groups = rig.bodies.map(() => { const g = new THREE.Group(); scene.add(g); return g; });
      for (const g of rig.geometry) {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(g.v), 3));
        geo.computeVertexNormals();
        const col = new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]);
        const m = new THREE.Mesh(geo, mat(col.getHex(), { flatShading: true, roughness: 0.7 }));
        m.castShadow = true; m.receiveShadow = true;
        m.userData.base = col.getHex();
        groups[g.body].add(m);
      }
      objs.bodies = groups;
      objs.feet = rig.feet.map(i => groups[i]);
    } else if (rig.kind === 'arm') {
      // static base column (the kernel's body 1 is a decoupled phantom; base_link is welded to the world)
      const base = new THREE.Mesh(new THREE.CylinderGeometry(rig.base.radius, rig.base.radius * 1.25, rig.base.height, 40), mat(0x3f4852, { roughness: 0.8 }));
      base.rotation.x = Math.PI / 2; base.position.z = rig.base.height / 2; base.castShadow = true; base.receiveShadow = true; scene.add(base);
      const plate = new THREE.Mesh(new THREE.CylinderGeometry(rig.base.radius * 1.5, rig.base.radius * 1.5, 0.04, 48), mat(0x2f363f));
      plate.rotation.x = Math.PI / 2; plate.position.z = 0.02; plate.receiveShadow = true; scene.add(plate);
      const shades = [0xb9c2cc, 0xa7b1bc, 0x98a3af, 0x8a95a2, 0x7c8794, 0x6e7986];
      const Y = new THREE.Vector3(0, 1, 0);
      const bodies = new Array(rig.links.length + 2).fill(null);
      rig.links.forEach((l, j) => {
        const g = new THREE.Group();
        const p0 = new THREE.Vector3(...l.p0), p1 = new THREE.Vector3(...l.p1);
        const dir = p1.clone().sub(p0); const len = dir.length();
        if (len > 1e-6) {
          const cyl = new THREE.Mesh(new THREE.CylinderGeometry(l.radius, l.radius * 0.85, len, 28), mat(shades[j]));
          cyl.position.copy(p0).addScaledVector(dir, 0.5);
          cyl.quaternion.setFromUnitVectors(Y, dir.clone().normalize());
          cyl.castShadow = true; cyl.receiveShadow = true; g.add(cyl);
        }
        const hub = new THREE.Mesh(new THREE.SphereGeometry(l.radius * 1.08, 28, 20), mat(0x55606c, { roughness: 0.5 }));
        hub.position.copy(p0); hub.castShadow = true; g.add(hub);
        scene.add(g); bodies[l.body] = g;
      });
      objs.bodies = bodies;
      // tip marker + target
      objs.tip = new THREE.Mesh(new THREE.SphereGeometry(0.045, 20, 14), mat(0x3987e5, { emissive: 0x1c5cab, emissiveIntensity: 0.6 }));
      scene.add(objs.tip);
      const tr = Math.max(rig.acq_radius_m, 0.05);
      objs.target = new THREE.Mesh(new THREE.SphereGeometry(tr, 24, 16), mat(0xd03b3b, { emissive: 0xd03b3b, emissiveIntensity: 0.35, transparent: true, opacity: 0.92 }));
      scene.add(objs.target);
      objs.targetHalo = new THREE.Mesh(new THREE.SphereGeometry(tr * 2.2, 24, 16), new THREE.MeshBasicMaterial({ color: 0xd03b3b, transparent: true, opacity: 0.12, depthWrite: false }));
      scene.add(objs.targetHalo);
      objs.acquired = new THREE.Group(); scene.add(objs.acquired);
      objs.acqMat = mat(0x0ca30c, { emissive: 0x0ca30c, emissiveIntensity: 0.4, transparent: true, opacity: 0.75 });
      objs.acqGeo = new THREE.SphereGeometry(tr * 0.8, 16, 12);
      // tier ball around the home tip (faint) + judge proxies
      const tier = clipMeta && clipMeta.tier != null ? clipMeta.tier : 0;
      const ball = new THREE.Mesh(new THREE.SphereGeometry(rig.tier_radius_m[tier], 36, 24), new THREE.MeshBasicMaterial({ color: 0x3987e5, wireframe: true, transparent: true, opacity: 0.07, depthWrite: false }));
      ball.position.set(...rig.home_tip); scene.add(ball); objs.tierBall = ball;
      const pr = rig.proxies;
      const col = new THREE.Mesh(new THREE.CylinderGeometry(pr.column_radius_m, pr.column_radius_m, pr.column_height_m, 48, 1, true), new THREE.MeshBasicMaterial({ color: 0xec835a, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }));
      col.rotation.x = Math.PI / 2; col.position.z = pr.column_height_m / 2; scene.add(col);
      const fm = new THREE.Mesh(new THREE.PlaneGeometry(gsz, gsz), new THREE.MeshBasicMaterial({ color: 0xec835a, transparent: true, opacity: 0.05, side: THREE.DoubleSide, depthWrite: false }));
      fm.position.z = pr.floor_margin_m; scene.add(fm);
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineDashedMaterial({ color: 0xfab219, dashSize: 0.05, gapSize: 0.04, transparent: true, opacity: 0.8 }));
      scene.add(line); objs.tipLine = line;
    } else {
      // generic fallback: small boxes for each body pose
      const n = (rig.bodies || []).length || 16;
      for (let i = 0; i < n; i++) { const m = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.03, 0.03), mat(0x9aa5b1)); m.castShadow = true; scene.add(m); objs.bodies.push(m); }
    }
    T.scene = scene; T.objs = objs; T.rigKey = rigKey; T.rig = rig;
    // camera
    const cam = T.camera, ctl = T.controls;
    if (rig.kind === 'arm') { ctl.target.set(1.2, 0, 1.4); cam.position.set(1.2 + 4.2, -5.6, 3.6); }
    else if (rigKey === 'duck') { ctl.target.set(0, 0, 0.13); cam.position.set(0.55, -0.9, 0.42); }
    else { ctl.target.set(0, 0, 0.9); cam.position.set(2.9, -3.4, 1.9); }
    ctl.update();
    P.lastRootX = null;
  }

  function applyFrame(f) {
    const clip = P.clip, rig = T.rig, objs = T.objs;
    if (!clip || !objs) return;
    const fr = clip.frames[f];
    let rootX = 0;
    if (rig.kind === 'arm') {
      fr.bodies.forEach((s, i) => { const o = objs.bodies[i]; if (!o) return; o.position.set(s[0], s[1], s[2]); o.quaternion.set(s[3], s[4], s[5], s[6]); });
      objs.tip.position.set(...fr.tip);
      const tg = clip.targets[f]; objs.target.position.set(...tg); objs.targetHalo.position.copy(objs.target.position);
      const hold = clip.hold_steps[f];
      const acqHere = clip.acquired_steps.includes(f);
      const c = acqHere ? 0x0ca30c : hold > 0 ? 0xfab219 : 0xd03b3b;
      objs.target.material.color.setHex(c); objs.target.material.emissive.setHex(c); objs.targetHalo.material.color.setHex(c);
      objs.targetHalo.material.opacity = hold > 0 ? 0.12 + 0.18 * Math.min(1, hold / 14) : 0.1;
      // acquired targets so far (persistent green markers)
      const want = clip.acquired_steps.filter(s => s <= f);
      while (objs.acquired.children.length > want.length) objs.acquired.remove(objs.acquired.children[objs.acquired.children.length - 1]);
      while (objs.acquired.children.length < want.length) {
        const s = want[objs.acquired.children.length];
        const m = new THREE.Mesh(objs.acqGeo, objs.acqMat); m.position.set(...clip.targets[s - 1]); objs.acquired.add(m);
      }
      const pos = objs.tipLine.geometry.attributes.position; pos.setXYZ(0, ...fr.tip); pos.setXYZ(1, ...tg); pos.needsUpdate = true; objs.tipLine.computeLineDistances();
      const d = Math.hypot(fr.tip[0] - tg[0], fr.tip[1] - tg[1], fr.tip[2] - tg[2]);
      const idx = clip.target_index[f];
      const acquiredN = want.length;
      hud([
        chip(`t ${(f * clip.dt).toFixed(2)} s`),
        chip(`tier ${clip.tier} · seed ${clip.seed}`),
        chip(`targets ${acquiredN}/5`, acquiredN === 5 ? 'acq' : ''),
        chip(`target #${Math.min(idx + 1, 5)} · ${(d * 100).toFixed(1)} cm`, d <= clip.acq_radius_m ? 'on' : ''),
        chip(`hold ${hold}/14`, hold > 0 ? 'warn' : ''),
      ]);
    } else {
      fr.forEach((s, i) => { const o = objs.bodies[i]; if (!o) return; o.position.set(s[0], s[1], s[2]); o.quaternion.set(s[3], s[4], s[5], s[6]); });
      rootX = fr[1][0];
      const ct = clip.contacts[f] || [false, false];
      objs.feet.forEach((o, k) => setHighlight(o, ct[k]));
      const t = f * clip.dt;
      const ff = clip.footfalls.filter(x => x.t <= t + 1e-9);
      const L = ff.filter(x => x.foot === 'left').length, R = ff.length - L;
      const tilt = tiltDeg(fr[1], T.rigKey);
      hud([
        chip(`t ${t.toFixed(2)} s`),
        chip(`cmd ${clip.command.toFixed(2)} m/s`),
        chip(`x ${rootX.toFixed(2)} m`),
        chip(`L ${ct[0] ? 'down' : 'up'}`, ct[0] ? 'on' : ''),
        chip(`R ${ct[1] ? 'down' : 'up'}`, ct[1] ? 'on' : ''),
        chip(`qualified footfalls ${ff.length} (L${L}/R${R})`, ff.length ? 'acq' : ''),
        chip(`tilt ${tilt.toFixed(1)}°`),
      ]);
    }
    // follow camera
    if (V.follow.checked && rig.kind !== 'arm') {
      if (P.lastRootX == null) P.lastRootX = rootX;
      const dx = rootX - P.lastRootX;
      T.controls.target.x += dx; T.camera.position.x += dx;
      T.objs.sun.position.x += dx; T.objs.sun.target.position.x += dx;
      P.lastRootX = rootX;
    }
    const ended = (clip.fell_at || clip.ended_at) && f >= clip.frames.length - 1;
    V.fell.style.display = ended ? 'block' : 'none';
    V.fell.textContent = clip.fell_at ? `episode terminated at ${clip.fell_at} s (fall / tilt)` : `episode ended at ${clip.ended_at} s (proxy violation)`;
    V.time.textContent = `${(f * clip.dt).toFixed(2)} / ${((clip.frames.length - 1) * clip.dt).toFixed(2)} s`;
    V.scrub.value = f;
  }
  function setHighlight(o, on) {
    const apply = m => { if (!m.material) return; if (on) { m.material.color.setHex(0x2bd39a); m.material.emissive.setHex(0x0f6e4c); m.material.emissiveIntensity = 0.55; } else { m.material.color.setHex(m.userData.base); m.material.emissive.setHex(0x000000); } };
    if (o.isMesh) apply(o); else o.traverse(ch => { if (ch.isMesh) apply(ch); });
  }
  function tiltDeg(s, rigKey) {
    // body up axis: humanoid bodies are authored y-up (local +Y), duck is z-up (local +Z)
    const q = new THREE.Quaternion(s[3], s[4], s[5], s[6]);
    const up = (rigKey === 'duck') ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
    up.applyQuaternion(q);
    return Math.acos(Math.min(1, Math.max(-1, up.z))) * 180 / Math.PI;
  }
  function chip(text, cls) { return el('span', { class: 'chip ' + (cls || '') }, text); }
  function hud(chips) { V.hud.replaceChildren(...chips); }

  function render(force) {
    if (!T.renderer || !T.scene) return;
    T.controls.update();
    T.renderer.render(T.scene, T.camera);
  }
  function loop(ts) {
    P.raf = requestAnimationFrame(loop);
    if (!overlay.classList.contains('open') || !P.clip) return;
    const dt = Math.min(0.1, (ts - (P.last || ts)) / 1000); P.last = ts;
    if (P.playing) {
      P.acc += dt * P.speed;
      let adv = 0;
      while (P.acc >= P.clip.dt) { P.acc -= P.clip.dt; adv++; }
      if (adv) {
        const n = P.clip.frames.length;
        let nf = P.frame + adv;
        if (nf >= n) { nf = n - 1; P.holdEnd = (P.holdEnd || 0) + dt; if (P.holdEnd > 1.2) { nf = 0; P.holdEnd = 0; P.lastRootX = null; } }
        if (nf !== P.frame) { P.frame = nf; applyFrame(P.frame); }
      }
    }
    render();
  }
  function step(d) { if (!P.clip) return; P.frame = Math.max(0, Math.min(P.clip.frames.length - 1, P.frame + d)); applyFrame(P.frame); render(); }
  function setPlaying(p) { P.playing = p; V.play.textContent = p ? 'Pause' : 'Play'; }
  function setSpeed(s) { P.speed = s; for (const b of V.speed.children) b.classList.toggle('active', +b.dataset.s === s); }

  async function openViewer(cardId, clipId) {
    const card = M.cards.find(c => c.id === cardId);
    if (!card) return;
    P.card = card;
    overlay.classList.add('open');
    document.body.style.overflow = 'hidden';
    ensureRenderer();
    V.title.textContent = `${card.robot} — ${card.title}`;
    const st = STATUS[card.status] || STATUS.candidate;
    V.badge.className = `badge ${card.status}`; V.badge.textContent = `${st[1]} ${st[0]} · ${card.verdict.label}`;
    renderSide(card, null);
    const meta = card.clips.find(c => c.id === clipId) || card.clips[0];
    if (!meta) { V.hud.replaceChildren(chip('no clips recorded for this card')); return; }
    await selectClip(meta);
    if (!P.raf) P.raf = requestAnimationFrame(loop);
  }
  async function selectClip(meta) {
    P.meta = meta;
    location.hash = `card=${encodeURIComponent(P.card.id)}&clip=${encodeURIComponent(meta.id)}`;
    for (const b of V.side.querySelectorAll('.clipbtn')) b.classList.toggle('active', b.dataset.id === meta.id);
    V.hud.replaceChildren(chip('loading clip…'));
    let rig = M.rigs[meta.rig];
    if (rig.kind === 'mesh-external') rig = await load('rig:duck', rig.file);
    const clip = await load('clip:' + meta.id, meta.file);
    const rigChanged = T.rigKey !== meta.rig || !T.scene || (rig.kind === 'arm' && T.objs && T.objs.tierBall && P.clip && P.clip.tier !== clip.tier);
    if (rigChanged) buildScene(meta.rig, rig, clip);
    P.clip = clip; P.frame = 0; P.acc = 0; P.lastRootX = null; P.holdEnd = 0;
    V.scrub.max = clip.frames.length - 1;
    V.legend.innerHTML = rig.kind === 'arm'
      ? 'target: <span style="color:#f39a9a">red</span> → <span style="color:#ffd27a">in radius (holding)</span> → <span style="color:#8fe08f">acquired</span> · blue: flange tip · orange volumes: judge proxies · drag to orbit, wheel to zoom'
      : 'feet turn <span style="color:#8fe6c2">green</span> on contact · drag to orbit, wheel to zoom, right-drag to pan';
    renderSide(P.card, clip);
    setPlaying(true);
    resize();
    applyFrame(0);
    render(true);
  }
  function closeViewer() {
    overlay.classList.remove('open');
    document.body.style.overflow = '';
    if (location.hash) history.replaceState(null, '', location.pathname);
  }

  // ------------------------------------------------------------ side panel
  function renderSide(card, clip) {
    const side = V.side; side.replaceChildren();
    // clip selector, grouped
    const groups = {};
    for (const m of card.clips) {
      const g = card.family === 'arm' ? `seed ${m.seed}` : (M.dr_labels[m.dr] || m.dr || 'nominal');
      (groups[g] = groups[g] || []).push(m);
    }
    const sel = el('div', null, el('h4', null, `Clips (${card.clips.length})`));
    for (const [g, ms] of Object.entries(groups)) {
      sel.append(el('div', { class: 'clipgroup' }, Object.keys(groups).length > 1 ? el('div', { class: 'gname' }, g) : null,
        el('div', { class: 'clipbtns' }, ms.map(m => el('button', { class: 'clipbtn' + (P.meta && P.meta.id === m.id ? ' active' : ''), 'data-id': m.id, onclick: () => selectClip(m) },
          el('span', { class: 'st ' + (m.verdict && m.verdict.passed ? 'p' : 'f') }),
          card.family === 'arm' ? `tier ${m.tier}` : `${m.command.toFixed(2)} m/s`)))));
    }
    side.append(sel);
    // current clip verdict
    if (clip) {
      const v = clip.verdict;
      const kv = el('div', { class: 'kv' });
      const add = (k, val, mono) => kv.append(el('span', { class: 'k' }, k), el('span', { class: 'v' + (mono ? ' mono' : '') }, val));
      if (card.family === 'arm') {
        add('Judge', v.passed ? 'PASS' : 'fail');
        add('Acquired', `${v.acquired}/5 targets`);
        add('Times', v.acquisition_times_s.map(t => t == null ? '—' : t.toFixed(2) + ' s').join(', '));
        add('Max speed ratio', v.max_speed_ratio == null ? '—' : v.max_speed_ratio.toFixed(3) + ' × URDF limit');
        add('Tier radius', `${clip.tier_radius_m} m around the home tip`);
        if (clip.ended_at) add('Ended', `${clip.ended_at} s (proxy)`);
      } else {
        add('Judge', v.passed ? 'PASS' : 'fail');
        add('Qualified footfalls', `${v.qualified} (L ${v.left} / R ${v.right})`);
        add('Distance', `${v.distance_m} m of ${(clip.command * 8).toFixed(2)} m commanded`);
        add('Alive', `${v.alive_s} s` + (clip.fell_at ? ` (fell at ${clip.fell_at} s)` : ''));
        if (clip.dr && clip.dr !== 'nominal') add('Pinned dynamics', Object.entries(clip.dr_pins).map(([k, x]) => `${k}=${typeof x === 'number' ? +x.toFixed(4) : x}`).join(', '), true);
      }
      add('Seed', `${clip.seed}`);
      add('Physics', clip.physics, true);
      const crit = el('ul', { class: 'crit' }, v.failed_criteria.length ? v.failed_criteria.map(f => el('li', { class: 'fail' }, `✕ ${f}`)) : el('li', null, '✓ all criteria pass'));
      side.append(el('div', null, el('h4', null, 'This clip · frozen judge'), kv, crit));
    }
    // acceptance cell grid
    const cells = card.verdict.cells || [];
    if (cells.length) {
      const cols = card.family === 'arm' ? ['tier 0', 'tier 1', 'tier 2'] : (card.family === 'duck' ? ['0.10', '0.15', '0.20'] : ['0.50', '0.75', '1.00']);
      const seeds = [...new Set(cells.map(c => c.cell.split('-')[0]))];
      const grid = el('div', { class: 'cellgrid' }, el('span'), cols.map(c => el('span', { class: 'h' }, c)));
      for (const s of seeds) {
        grid.append(el('span', { class: 's' }, s.replace('seed', 's')));
        for (let i = 0; i < 3; i++) {
          const c = cells.filter(x => x.cell.startsWith(s + '-'))[i];
          if (!c) { grid.append(el('span', { class: 'c' }, '—')); continue; }
          const cur = clip && (card.family === 'arm' ? (c.cell === `seed${clip.seed}-tier${clip.tier}`) : (c.cell === `seed${clip.seed}-cmd${clip.command.toFixed(2)}`));
          const label = card.family === 'arm' ? (c.acquired == null ? (c.passed ? 'PASS' : 'fail') : `${c.acquired}/5`) : `q${c.qualified}`;
          grid.append(el('span', { class: 'c ' + (c.passed ? 'p' : 'f') + (cur ? ' cur' : ''), title: `${c.cell}: ${c.passed ? 'PASS' : 'fail — ' + (c.failed || []).join(', ')}` }, label));
        }
      }
      const v = card.verdict;
      side.append(el('div', null, el('h4', null, `Acceptance ${v.passed}/${v.total} · ${card.family === 'arm' ? 'seeds × tiers' : 'seeds × commands'}`), grid,
        el('div', { style: 'color:var(--text-muted);font-size:11.5px;margin-top:6px' },
          v.source || '', v.committed ? ` · committed ${v.committed.path.split('/').slice(-1)[0]}: ${v.committed.passed}/${v.committed.total}${v.committed.matches ? ' (matches)' : ' (DIFFERS)'}` : '')));
    }
    // brittleness (generalist)
    if (card.brittleness) {
      const b = card.brittleness;
      const box = el('div', { class: 'britt' });
      for (const r of b.rows) {
        const hi = clip && r.config === clip.dr;
        box.append(el('div', { class: 'row' + (hi ? ' hi' : '') },
          el('span', { class: 'n' }, r.config, ' ', el('small', null, r.pins === 'authored' ? '' : r.pins.replace(/_scale/g, ''))),
          el('span', { class: 'mono' }, `${r.passed}/${r.total}`),
          el('span', { class: 'bar' }, el('i', { style: `width:${100 * r.passed / r.total}%` }))));
      }
      side.append(el('div', null, el('h4', null, `Brittleness ${b.passed}/${b.total} · seeds 4242 & 7 × 3 commands`), box));
    }
    // paths
    side.append(el('div', null, el('h4', null, 'Evidence'),
      el('div', { class: 'paths' },
        el('div', null, el('b', null, 'actor '), card.actor, card.actor_sha256 ? ` (sha256 ${card.actor_sha256.slice(0, 12)}…, ${fmt(card.actor_bytes / 1e3, 0)} kB)` : ''),
        el('div', null, el('b', null, 'evidence '), card.evidence),
        el('div', null, el('b', null, 'judge '), card.judge),
        el('div', { style: 'margin-top:6px;color:var(--text-secondary);font-family:var(--sans);font-size:12px' }, card.lineage))));
  }

  // deep link
  function fromHash() {
    const h = new URLSearchParams(location.hash.replace(/^#/, ''));
    if (h.get('card')) openViewer(h.get('card'), h.get('clip'));
  }
  fromHash();
  window.addEventListener('hashchange', () => { if (location.hash && !overlay.classList.contains('open')) fromHash(); });
})();

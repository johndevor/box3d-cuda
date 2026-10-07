"""Tech-tree curriculum controller: one run that senses which walking
requirements are met and advances its own difficulty.

The ladder is DATA (a list of Stage rows, shippable as JSON): each stage
carries the runtime gate-termination knobs and command distribution it
enforces on entry, plus the predicate over recent gate_proxy metrics that
unlocks the next node. All compute goes to the current frontier: early
stages let the population learn to exist, later stages terminate
judge-failing behavior at runtime (dwc1_set_gate_termination, commit
65d99b8) so gradient never subsidizes it.

The controller is intentionally dumb and auditable:
- it reads ONLY the per-update metrics dict gpu_train already writes
  (gate_proxy_ep_qualified_l/r, gate_proxy_ep_alt_violations,
  ep_len_mean) -- no new instrumentation, no reward coupling;
- it acts ONLY at update boundaries, ONLY through the two public lane
  knobs (set_gate_termination, per-reset command override) -- the physics
  and reward stay bit-identical within a stage;
- ADVANCE: the stage's metric median over its trailing window meets the
  threshold (windows reset on every transition; a stage must also dwell
  at least its window length before advancing, so one lucky update can't
  skip a node);
- DE-ESCALATE (the fail-back-down edge every tech tree needs): if median
  ep_len collapses below the stage's floor for `collapse_k` CONSECUTIVE
  updates, step back ONE stage -- an enforcement knob that kills the
  whole population leaves no gradient to learn from, so the run backs off
  to the last survivable node and re-earns the advance;
- never advances past the terminal stage; never de-escalates below the
  first.

Ship ladders: "humanoid-walk" (HUMANOID_WALK_STAGES below) and
"arm-reach" (ARM_REACH_STAGES: the same two lane knobs in their reach
semantics -- first-acquisition deadline / judge-clause violating-tick
cap -- over the reach mapping of gate_proxy), both robot-guarded.
No duck ladder is authored -- requesting a curriculum for the duck is an
explicit error, and gpu_train without --curriculum is byte-identical to
the pre-curriculum trainer (fingerprint-proven).
"""
from __future__ import annotations

import dataclasses
import json
import statistics
from pathlib import Path

# Judge clause anchors (walk/eval/humanoid_gait.py, FROZEN): FIRST_STEP_S
# 2.5 s, MIN_FOOTFALLS 6 (>= 3/foot), strict alternation, 8 s episodes.
# Ticks are 0.002 s: 2.5 s = 1250 ticks, 5 s = 2500 ticks.


@dataclasses.dataclass(frozen=True)
class Stage:
    name: str
    # -- actions_on_enter ---------------------------------------------------
    first_deadline_ticks: int = 0        # 0 = off (lane knob)
    max_alternation_violations: int = 0  # 0 = off (lane knob)
    commands: tuple | None = None        # per-reset command pool; None = env's
    # -- advance_when (None metric = terminal stage) -------------------------
    advance_metric: str | None = None    # key derived from the metrics row
    advance_threshold: float = 0.0       # median(metric over window) >= this
    advance_window: int = 8              # K updates (also the minimum dwell)
    # optional SECOND predicate (OR): median(advance_metric2) <= threshold2
    # when advance_below2 else >= -- e.g. "the stage's own median violating
    # ticks are already inside the judge's tolerance" (arm ladder)
    advance_metric2: str | None = None
    advance_threshold2: float = 0.0
    advance_below2: bool = False
    # -- de-escalation -------------------------------------------------------
    collapse_ep_len: float = 25.0        # median ep_len floor (policy steps)
    collapse_k: int = 5                  # consecutive updates below the floor
    note: str = ""


# Derived metrics available to predicates (from the gpu_train row):
#   qualified_total = gate_proxy_ep_qualified_l + gate_proxy_ep_qualified_r
#   alt_violations  = gate_proxy_ep_alt_violations
#   ep_len          = ep_len_mean (None -> +inf: no episode ENDED, nobody died)
#   acq_per_live_s  = qualified_total / (ep_len_mean * CONTROL_DT): episode-
#                     LENGTH-NORMALISED evidence. A stage whose enforcement
#                     knob shortens episodes (a violating-tick cap kills at
#                     3-4 s) also caps the raw per-episode count, so a raw
#                     qualified_total threshold can be structurally
#                     unreachable inside that stage (the arm's
#                     deadline_cap_40 sat at median 0.41 acq/episode over
#                     ~3.4 s live = 0.12 acq/s while needing 2.0/episode).
#                     None when no episode has ended (no evidence).
CONTROL_DT_S = 0.02


def _derive(line: dict) -> dict:
    ql = line.get("gate_proxy_ep_qualified_l")
    qr = line.get("gate_proxy_ep_qualified_r")
    ep = line.get("ep_len_mean")
    total = None if ql is None or qr is None else float(ql) + float(qr)
    return {
        "qualified_total": total,
        "alt_violations": (None if line.get("gate_proxy_ep_alt_violations")
                           is None
                           else float(line["gate_proxy_ep_alt_violations"])),
        "ep_len": float("inf") if ep is None else float(ep),
        "acq_per_live_s": (None if total is None or not ep
                           else total / (float(ep) * CONTROL_DT_S)),
    }


# ---------------------------------------------------------------------------
# The humanoid walking ladder. Threshold rationales reference the FROZEN
# judge (walk/eval/humanoid_gait.py) clause the stage is scaffolding.
HUMANOID_WALK_STAGES = (
    Stage(
        name="free",
        commands=(0.75,),
        advance_metric="qualified_total", advance_threshold=0.5,
        note="No enforcement; single mid command concentrates learning on "
             "one cadence. Advance when qualified swings EXIST at all "
             "(median total >= 0.5/episode): the gate_proxy 'qualified "
             "swing' already shadows the judge's duration+clearance+"
             "placement clauses, so 0.5 means real steps are appearing."),
    Stage(
        name="swings_appear",
        commands=(0.75,),
        advance_metric="qualified_total", advance_threshold=2.0,
        note="Still nothing enforced, just logged (the coordinator's "
             "observation node). Advance at median >= 2: one qualified "
             "swing per foot -- the minimal alternating unit on the road "
             "to the judge's 6-with-3-per-foot."),
    Stage(
        name="deadline_loose",
        first_deadline_ticks=2500,       # 5 s = 2x the judge's clause
        commands=(0.50, 0.75),
        advance_metric="qualified_total", advance_threshold=3.0,
        note="First enforcement: prune never-steppers at 5 s -- double the "
             "judge's 2.5 s FIRST_STEP_S so slow starters still get "
             "gradient. Second command joins (slow walking = longest "
             "balance demands; the duck oversampled it for the same "
             "reason). Advance at median >= 3 qualified swings."),
    Stage(
        name="deadline_judge",
        first_deadline_ticks=1250,       # the judge's 2.5 s, exactly
        commands=(0.50, 0.75, 1.00),
        advance_metric="qualified_total", advance_threshold=4.0,
        note="Judge-exact first-step deadline (FIRST_STEP_S 2.5 s -> 1250 "
             "ticks); full command distribution -- the judge scores all "
             "three. Advance at median >= 4 (two alternating pairs)."),
    Stage(
        name="alternation_cap",
        first_deadline_ticks=1250,
        max_alternation_violations=3,
        commands=(0.50, 0.75, 1.00),
        advance_metric="qualified_total", advance_threshold=6.0,
        note="Kill persistent limps: 3 consecutive same-foot qualified "
             "touchdowns is a limp the judge's footfalls_alternate clause "
             "would reject outright; the cap of 3 still tolerates brief "
             "doubles while learning. Advance at median >= 6 = the "
             "judge's MIN_FOOTFALLS."),
    Stage(
        name="full",
        first_deadline_ticks=1250,
        max_alternation_violations=1,
        commands=(0.50, 0.75, 1.00),
        advance_metric=None,             # terminal node
        note="Judge-tight everything: 2.5 s deadline + ANY repeated-foot "
             "qualified touchdown terminates (the judge requires strict "
             "alternation). Surviving 8 s under these knobs is, by "
             "construction, close to a judge-passing episode; the frozen "
             "CPU judge remains the only acceptance authority."),
)

# ---------------------------------------------------------------------------
# The ARM reach ladder (kernel ABI v8 REACH kind). The two lane knobs carry
# reach semantics (duck_cuda.h): first_deadline_ticks = no target ACQUIRED
# within that many accepted ticks; max_alternation_violations = the
# frozen judge's clause 3/4/5 VIOLATING TICKS (joint limit / joint speed /
# floor-column proxy, counted per accepted tick) reaching that count. The
# metrics are the same gate_proxy_* rows: qualified_total = targets
# acquired per episode (qualified_right is 0 on the reach mapping),
# alt_violations = violating ticks per episode. Judge anchors
# (walk/eval/arm_reach_judge.py, FROZEN): 5 targets in 8 s = 4000 ticks,
# every tick inside the limits/speeds/proxy, i.e. ZERO violating ticks.
# SELF-GATING on episode-length-normalised evidence. Every cap/deadline
# stage advances on `acq_per_live_s` = acquisitions per live SECOND, so a
# stage whose knob shortens episodes cannot suppress its own advance
# metric (the raw acq/episode rule never advanced past deadline_cap_40 in
# runs/gpu/20260902-160543-arm-reach-kr240: median 0.41/episode over 3.4 s
# = 0.12 acq/s, needing 2.0). Ramp toward the judge: a passing episode is
# 5 acquisitions in <= 8 s = 0.625 acq/s. Thresholds re-derived from the
# measured stage medians of that GPU run and the local 128-env delta run
# (acq/s medians: free 0.000, cap_2000 0.015-0.029, cap_800 0.025-0.070,
# cap_200 0.079-0.113, deadline_cap_40 0.116-0.121 with trailing-8 medians
# reaching 0.15-0.20). The cap stages ALSO advance when their own median
# violating ticks/episode is already inside the judge's tolerance (<= 5:
# a single 100 ms excursion), the evidence the cap knob exists to create.
# Offline replay of the GPU run's metrics through this rule advances
# deadline_cap_40 at update 94 (arm/tests/test_arm_curriculum.py).
ARM_REACH_STAGES = (
    Stage(
        name="free",
        advance_metric="acq_per_live_s", advance_threshold=0.006,
        collapse_ep_len=100.0,
        note="No enforcement. Advance when acquisitions EXIST at all "
             "(median >= 0.006 acq/s = 0.05 per 8 s episode over 8 "
             "updates): the acquisition counter IS the judge's clause-2 "
             "rule (14 consecutive in-radius boundaries), so any nonzero "
             "median means the hold behavior appeared under exploration."),
    Stage(
        name="violations_cap_2000",
        max_alternation_violations=2000,  # 50 % of the 4000 ticks
        advance_metric="acq_per_live_s", advance_threshold=0.03,
        advance_metric2="alt_violations", advance_threshold2=5.0,
        advance_below2=True,
        collapse_ep_len=100.0,
        note="First enforcement, just under the measured noisy-policy "
             "baseline of the ABS lineage (~2300 violating ticks/episode): "
             "episodes that spend more than half their ticks outside the "
             "judge's limit/speed/proxy clauses terminate. Advance at "
             "median >= 0.03 acq/s (0.25 per 8 s) or violating ticks <= 5."),
    Stage(
        name="violations_cap_800",
        max_alternation_violations=800,   # 20 %
        advance_metric="acq_per_live_s", advance_threshold=0.06,
        advance_metric2="alt_violations", advance_threshold2=5.0,
        advance_below2=True,
        collapse_ep_len=100.0,
        note="Halve the tolerated violating ticks again. Advance at median "
             ">= 0.06 acq/s (0.5 per 8 s) or violating ticks <= 5."),
    Stage(
        name="violations_cap_200",
        max_alternation_violations=200,   # 5 %
        advance_metric="acq_per_live_s", advance_threshold=0.10,
        advance_metric2="alt_violations", advance_threshold2=5.0,
        advance_below2=True,
        collapse_ep_len=100.0,
        note="5 % violating ticks (0.4 s of overspeed per episode). "
             "Advance at median >= 0.10 acq/s (0.8 per 8 s; the GPU run's "
             "stage median was 0.113) or violating ticks <= 5."),
    Stage(
        name="deadline_cap_40",
        first_deadline_ticks=1500,        # 3 s to the FIRST acquisition
        max_alternation_violations=40,    # 1 %
        advance_metric="acq_per_live_s", advance_threshold=0.15,
        advance_metric2="alt_violations", advance_threshold2=5.0,
        advance_below2=True,
        collapse_ep_len=100.0,
        note="Tighten to 1 % violating ticks (a single 80 ms overspeed) "
             "and prune never-acquirers: the judge needs 5 acquisitions in "
             "8 s (1.6 s each on average; the IK baseline's first lands at "
             "~1.1 s, the delta actor's at 0.6-1.2 s), so 3 s without a "
             "first acquisition is a lost episode. Advance at median >= "
             "0.15 acq/s (the GPU run's trailing-8 median crossed it at "
             "u94; its raw 0.41/episode never reached the old 2.0) or "
             "violating ticks <= 5 (the local delta run's stage median "
             "was 4.1)."),
    Stage(
        name="judge_tight",
        first_deadline_ticks=1000,        # 2 s
        max_alternation_violations=1,     # ZERO violating ticks tolerated
        advance_metric="acq_per_live_s", advance_threshold=0.5,
        collapse_ep_len=100.0,
        note="Judge-tight: the first violating tick of any clause "
             "terminates (the judge fails the episode on one), first "
             "acquisition within 2 s. Advance at median >= 0.5 acq/s "
             "(4 per 8 s -- one short of the judge's 5 in 8 s = 0.625)."),
    Stage(
        name="full",
        first_deadline_ticks=1000,
        max_alternation_violations=1,
        advance_metric=None,              # terminal node
        collapse_ep_len=100.0,
        note="Same judge-tight knobs, terminal. Surviving 8 s here with "
             ">= 5 acquisitions is, by construction, a judge-passing "
             "episode; the frozen CPU judge remains the only acceptance "
             "authority."),
)

BUILTIN_LADDERS = {"humanoid-walk": ("humanoid", HUMANOID_WALK_STAGES),
                   "arm-reach": ("arm", ARM_REACH_STAGES)}


def load_ladder(spec: str, robot: str):
    """Resolve --curriculum: builtin name or a JSON file path.

    JSON schema: {"robot": "<name>", "stages": [{Stage fields}...]}."""
    if spec in BUILTIN_LADDERS:
        ladder_robot, stages = BUILTIN_LADDERS[spec]
        if robot != ladder_robot:
            raise SystemExit(f"--curriculum {spec} is authored for robot "
                             f"{ladder_robot!r}, not {robot!r} (no ladder "
                             "exists for the duck)")
        return stages
    path = Path(spec)
    if not path.is_file():
        raise SystemExit(f"--curriculum {spec!r}: not a builtin "
                         f"({sorted(BUILTIN_LADDERS)}) and not a file")
    data = json.loads(path.read_text())
    if data.get("robot") != robot:
        raise SystemExit(f"{path} is authored for robot "
                         f"{data.get('robot')!r}, not {robot!r}")
    stages = tuple(Stage(**{**row, "commands":
                            tuple(row["commands"]) if row.get("commands")
                            else None})
                   for row in data["stages"])
    if not stages:
        raise SystemExit(f"{path}: empty stage list")
    return stages


class CurriculumController:
    """Update-boundary tech-tree walker over a Stage ladder."""

    def __init__(self, stages, quiet: bool = False):
        self.stages = tuple(stages)
        self.index = 0
        self.quiet = quiet
        self._window: list[dict] = []    # derived metrics since last change
        self._collapse_run = 0

    @property
    def stage(self) -> Stage:
        return self.stages[self.index]

    # -- knob application ----------------------------------------------------
    def apply(self, env) -> None:
        """Push the current stage's knobs to the env's lane. Requires a
        lane exposing set_gate_termination (guarded by gpu_train)."""
        stage = self.stage
        lane = getattr(env, "_lane", env)
        lane.set_gate_termination(stage.first_deadline_ticks,
                                  stage.max_alternation_violations)
        # per-reset command pool override (None restores the env's own
        # distribution); consumed by LanePolicyEnv.reset
        env.command_override = stage.commands

    # -- per-update sensing ----------------------------------------------------
    def observe(self, line: dict) -> dict | None:
        """Feed one metrics row; returns a transition event dict or None."""
        derived = _derive(line)
        self._window.append(derived)
        stage = self.stage
        update = int(line.get("update", -1))

        # de-escalation first: an all-dead population has no gradient
        if derived["ep_len"] < stage.collapse_ep_len:
            self._collapse_run += 1
        else:
            self._collapse_run = 0
        if self._collapse_run >= stage.collapse_k and self.index > 0:
            previous = stage.name
            self.index -= 1
            self._reset_window()
            return {"event": "stage", "direction": "de-escalate",
                    "name": self.stage.name, "from": previous,
                    "update": update,
                    "reason": f"median ep_len < {stage.collapse_ep_len} for "
                              f"{stage.collapse_k} consecutive updates"}

        # advance: median over the trailing window, full-dwell required
        if stage.advance_metric is None:
            return None                   # terminal: never past 'full'
        if len(self._window) < stage.advance_window:
            return None
        reason = self._predicate(stage.advance_metric, stage.advance_threshold,
                                 False, stage.advance_window)
        if reason is None and stage.advance_metric2 is not None:
            reason = self._predicate(stage.advance_metric2,
                                     stage.advance_threshold2,
                                     stage.advance_below2, stage.advance_window)
        if reason is not None:
            previous = stage.name
            self.index += 1
            self._reset_window()
            return {"event": "stage", "direction": "advance",
                    "name": self.stage.name, "from": previous,
                    "update": update, "reason": reason}
        return None

    def _predicate(self, metric: str, threshold: float, below: bool,
                   window: int) -> str | None:
        """Trailing-window median test; None when it does not hold (or the
        lane exposes no such metric yet)."""
        recent = [w[metric] for w in self._window[-window:]]
        if any(v is None for v in recent):
            return None
        med = statistics.median(recent)
        ok = med <= threshold if below else med >= threshold
        if not ok:
            return None
        return (f"median {metric} = {med:.3f} {'<=' if below else '>='} "
                f"{threshold} over {window} updates")

    def _reset_window(self) -> None:
        self._window.clear()
        self._collapse_run = 0

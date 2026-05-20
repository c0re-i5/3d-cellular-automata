"""refine.py — Deep refinement pipeline for marked discoveries.

Five passes per target discovery:

  A: Deep replay at higher resolution / longer horizon; harvest the
     simulator's own GPU debug stats (mean/std/min/max/active/COM/Rg/hist)
     into a timeseries matching `debug_runs/*.json`.
  B: Dynamics fingerprint from a handful of full voxel snapshots:
     period, translation, growth, clusters, symmetry, projection MI.
  C: Seed sensitivity — N reruns with different seeds; mean ± std of
     key end-state metrics.
  D: Parameter sensitivity — Latin-hypercube perturbation of free params;
     end-state metric per row.
  E: Neighbourhood map — k-NN in the same-rule discovery cloud (cosine
     in normalised-param space).

Layout:
    refinements/<rule>_<hash>/
        report.json        ← summary; consumed by GUI panel
        deep_replay.json   ← pass A timeseries (same schema as debug_runs/*)
        voxel_<step>.npy   ← 5 snapshots for pass B
        seed_sweep.json    ← pass C rows
        perturbation.json  ← pass D rows
        neighbours.json    ← pass E ranking
    refinements/.status/<hash>.json   ← live progress; polled by GUI

Parent discovery in discoveries.json gets a ``refinement`` block linking to
the report, plus a small ``key_metrics`` summary, and ``marked`` is cleared.

Usage:
    python refine.py --all-marked
    python refine.py --idx 1234
    python refine.py --hash abc1234567 [--rule game_of_life_3d]
    python refine.py --idx 1234 --size 96 --steps 1500 --seeds 8 --perturbations 20

Status file schema:
    {"hash": "...", "state": "running"|"done"|"failed",
     "pass": "A"|"B"|"C"|"D"|"E"|"writeback",
     "pass_pct": 0..1, "msg": "...",
     "started_at": iso, "updated_at": iso}
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from typing import Any

import numpy as np

from schema import get_field

# Headless GL context comes from Simulator(headless=True). Importing
# simulator.py is heavy (~3s) so we defer to inside main().

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DISC_PATH = os.path.join(THIS_DIR, 'discoveries.json')
REF_DIR = os.path.join(THIS_DIR, 'refinements')
STATUS_DIR = os.path.join(REF_DIR, '.status')


# ── small utilities ────────────────────────────────────────────────────

def _iso_now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def short_hash(entry: dict) -> str:
    """Stable 10-char id derived from (rule, sorted params, seed)."""
    rule = get_field(entry, 'rule', '')
    params = sorted((str(k), float(v)) for k, v in (get_field(entry, 'params', {}) or {}).items())
    seed = int(get_field(entry, 'seed', 0))
    key = json.dumps([rule, params, seed], sort_keys=True)
    return hashlib.sha1(key.encode('utf-8')).hexdigest()[:10]


def refinement_dir(entry: dict) -> tuple[str, str]:
    """Return (hash, full_path) for an entry's refinement directory."""
    h = short_hash(entry)
    safe_rule = ''.join(c if c.isalnum() or c in '-_' else '_' for c in get_field(entry, 'rule', 'unknown'))
    return h, os.path.join(REF_DIR, f"{safe_rule}_{h}")


# ── discovery JSON I/O with flock ──────────────────────────────────────

def _load_discoveries() -> list[dict]:
    if not os.path.exists(DISC_PATH):
        return []
    with open(DISC_PATH) as f:
        return json.load(f)


def _write_back_refinement(parent_index: int, refinement_block: dict) -> None:
    """Atomically update a single entry's ``refinement`` field and clear
    its ``marked`` flag. Uses flock on the canonical lock file."""
    lock_path = DISC_PATH + '.lock'
    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        with open(DISC_PATH) as f:
            data = json.load(f)
        if not (0 <= parent_index < len(data)):
            raise IndexError(f"parent_index {parent_index} out of range")
        data[parent_index]['refinement'] = refinement_block
        if data[parent_index].get('marked'):
            data[parent_index]['marked'] = False
            data[parent_index].pop('marked_at', None)
        tmp = DISC_PATH + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, DISC_PATH)


# ── status file (polled by GUI) ────────────────────────────────────────

class StatusWriter:
    def __init__(self, hash_: str):
        os.makedirs(STATUS_DIR, exist_ok=True)
        self.path = os.path.join(STATUS_DIR, f"{hash_}.json")
        self.hash = hash_
        self.started = _iso_now()
        self._state = 'running'
        self._pass = 'init'
        self._pct = 0.0
        self._msg = ''
        self._flush()

    def update(self, pass_: str | None = None, pct: float | None = None,
               msg: str | None = None, state: str | None = None) -> None:
        if pass_ is not None: self._pass = pass_
        if pct is not None: self._pct = max(0.0, min(1.0, float(pct)))
        if msg is not None: self._msg = msg
        if state is not None: self._state = state
        self._flush()

    def done(self, msg: str = 'done') -> None:
        self.update(state='done', pass_='done', pct=1.0, msg=msg)

    def fail(self, msg: str) -> None:
        self.update(state='failed', msg=msg)

    def _flush(self) -> None:
        body = {
            'hash': self.hash,
            'state': self._state,
            'pass': self._pass,
            'pass_pct': self._pct,
            'msg': self._msg,
            'started_at': self.started,
            'updated_at': _iso_now(),
        }
        try:
            tmp = self.path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(body, f, indent=2)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001  best-effort write
            pass  # status writes are best-effort


# ── headless simulator helpers ─────────────────────────────────────────

def _drain_debug_fence(sim, timeout: float = 2.0) -> None:
    t = time.time()
    while sim._debug_fence is not None and time.time() - t < timeout:
        sim._harvest_debug_stats()


def _force_debug_sample(sim) -> None:
    """Bypass the per-frame throttle and grab one stats sample."""
    sim._debug_steps_since_sample = sim._debug_sample_interval
    sim._dispatch_debug_stats()
    _drain_debug_fence(sim)


def _read_grid_float32(sim) -> np.ndarray:
    """Read the simulator's current grid into a (W,H,D,4) float32 array."""
    src = sim.tex_a if sim.ping == 0 else sim.tex_b
    raw = src.read()
    arr = np.frombuffer(raw, dtype=sim._tex_np_dtype).reshape(
        sim.size, sim.size, sim.size, 4)
    return arr.astype(np.float32, copy=True)


def _make_sim(rule: str, size: int, params: dict, dt: float | None,
              seed: int, init_variant: str | None):
    """Construct a headless Simulator and apply discovery state."""
    from simulator import Simulator  # heavy import
    sim = Simulator(size=size, rule=rule, headless=True)
    # Apply params (skip unknown keys silently — same policy as
    # simulator._load_discovery; warning would be noise in batch refine)
    for k, v in (params or {}).items():
        if k in sim.params:
            sim.params[k] = v
    if dt is not None:
        sim.dt = dt
    sim.seed = int(seed)
    if init_variant:
        # Only honour variants known to the rule, else fall back to default
        variants = sim.preset.get('init_variants') or []
        default_init = sim.preset.get('init')
        if init_variant == default_init or init_variant in variants:
            sim._current_init = init_variant
    sim._reset()
    return sim


def _release_sim(sim) -> None:
    """Best-effort GL teardown so the next refinement gets a fresh ctx."""
    try:
        sim.shutdown()  # may not exist
    except Exception:  # noqa: BLE001  cleanup hook, never fatal
        pass
    try:
        sim.ctx.release()
    except Exception:  # noqa: BLE001  GL resource release, never fatal
        pass


# ── Pass A: deep replay with debug-stats timeseries ────────────────────

def pass_a_deep_replay(entry: dict, *, size: int, steps: int,
                       sample_interval: int, status: StatusWriter,
                       out_dir: str) -> tuple[Any, list[tuple[int, np.ndarray]]]:
    """Run the discovery headless for ``steps`` steps at side ``size``,
    sampling debug stats every ``sample_interval`` steps. Also captures
    5 full voxel snapshots distributed across the run (start, 25%, 50%,
    75%, end) for pass B. Returns (sim, snapshots)."""
    status.update(pass_='A', pct=0.0, msg='building sim')

    sim = _make_sim(rule=entry['rule'], size=size,
                    params=get_field(entry, 'params', {}) or {},
                    dt=entry.get('dt'),
                    seed=int(get_field(entry, 'seed', 0)),
                    init_variant=entry.get('init_variant'))
    sim._debug_enabled = True
    sim._debug_sample_interval = max(1, int(sample_interval))

    # Snapshot schedule
    snap_steps = sorted({0, steps // 4, steps // 2, (3 * steps) // 4, steps - 1})
    snap_steps = [s for s in snap_steps if 0 <= s < steps]
    snapshots: list[tuple[int, np.ndarray]] = []

    next_report = time.time() + 1.0
    for i in range(steps):
        sim._step_sim()
        if (i % sim._debug_sample_interval) == 0:
            _drain_debug_fence(sim, 0.5)
            _force_debug_sample(sim)
        if i in snap_steps:
            grid = _read_grid_float32(sim)
            snapshots.append((i, grid))
            np.save(os.path.join(out_dir, f'voxel_{i:06d}.npy'),
                    grid.astype(np.float16))  # half-precision sidecars
        if time.time() > next_report:
            status.update(pct=(i + 1) / steps,
                          msg=f'replay {i + 1}/{steps}')
            next_report = time.time() + 0.5

    # Final sample
    _force_debug_sample(sim)
    status.update(pct=1.0, msg=f'replay {steps}/{steps}')

    # Persist timeseries in the same shape as `_debug_save_snapshot`
    # so the GUI debug-overlay's plot widgets can ingest it directly.
    deep = {
        'rule': entry['rule'],
        'params': get_field(entry, 'params', {}) or {},
        'dt': sim.dt,
        'seed': sim.seed,
        'init_variant': entry.get('init_variant') or sim._current_init,
        'size': sim.size,
        'steps': steps,
        'sample_interval': sim._debug_sample_interval,
        'history': list(sim._debug_history),
    }
    with open(os.path.join(out_dir, 'deep_replay.json'), 'w') as f:
        json.dump(deep, f, indent=2, default=_json_default)

    return sim, snapshots


def _json_default(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


# ── Pass B: dynamics fingerprint from voxel snapshots ──────────────────

def pass_b_fingerprint(snapshots: list[tuple[int, np.ndarray]],
                       debug_history: list[dict],
                       status: StatusWriter) -> dict:
    """Run grid-level analyses on the 5 voxel snapshots + the harvested
    debug-stats history. Returns a dict of named fingerprint metrics."""
    from ca_debug import analyses as A
    status.update(pass_='B', pct=0.0, msg='analyses')

    if not snapshots:
        return {'error': 'no snapshots'}

    grids = [g for _, g in snapshots]
    final = grids[-1]
    out: dict = {
        'snapshot_steps': [s for s, _ in snapshots],
        'structure': A.analyze_structure(final),
        'symmetry':  A.measure_symmetry(final),
        'clusters':  A.analyze_clusters(final),
        'projection_entropy':   A.projection_entropy(final),
        'projection_structure': A.projection_structure(final),
        'slice_gol_coherence':  A.slice_gol_coherence(final),
        'slice_mi':             A.slice_mutual_info(final),
        'spatial_variation':    A.spatial_variation(final),
    }
    status.update(pct=0.5, msg='dynamics')

    # Multi-snapshot detectors
    try:
        out['period']      = A.detect_period(grids)
    except Exception as e:  # noqa: BLE001  optional dynamics analysis
        out['period']      = {'error': str(e)}
    try:
        out['translation'] = A.detect_translation(grids)
    except Exception as e:  # noqa: BLE001  optional dynamics analysis
        out['translation'] = {'error': str(e)}

    # Growth needs metric_history. Build it from debug stats: alive_ratio
    # ≈ debug_history[i]['active_frac'].
    mh = [{'alive_ratio': s.get('active_frac', 0.0)} for s in debug_history]
    if mh:
        try:
            out['growth'] = A.detect_growth(mh)
        except Exception as e:  # noqa: BLE001  optional dynamics analysis
            out['growth'] = {'error': str(e)}

    # Summary stats from debug timeseries
    if debug_history:
        af = np.array([s.get('active_frac', 0.0) for s in debug_history], dtype=np.float64)
        rg = np.array([s.get('rg', np.nan) for s in debug_history], dtype=np.float64)
        rg = rg[~np.isnan(rg)]
        out['debug_summary'] = {
            'active_frac_mean':  float(af.mean()),
            'active_frac_std':   float(af.std()),
            'active_frac_min':   float(af.min()),
            'active_frac_max':   float(af.max()),
            'active_frac_final': float(af[-1]),
            'rg_mean': float(rg.mean()) if rg.size else None,
            'rg_std':  float(rg.std())  if rg.size else None,
            'n_samples': len(debug_history),
        }
    status.update(pct=1.0, msg='fingerprint done')
    return out


# ── Pass C: seed sensitivity ───────────────────────────────────────────

KEY_METRICS = ('active_frac_final', 'rg_final', 'finite_final')


def _end_state_metrics(sim) -> dict:
    """Pull a small set of end-state numbers from a finished sim."""
    snap = sim._debug_latest or {}
    return {
        'active_frac': float(snap.get('active_frac', 0.0)),
        'rg':          float(snap.get('rg') if snap.get('rg') is not None else 0.0),
        'finite_c0':   int((snap.get('finite') or [0])[0]),
        'nan_c0':      int((snap.get('nan') or [0])[0]),
        'inf_c0':      int((snap.get('inf') or [0])[0]),
        'mean_c0':     float((snap.get('mean') or [0.0])[0]),
        'std_c0':      float((snap.get('std')  or [0.0])[0]),
    }


def pass_c_seed_sensitivity(entry: dict, *, size: int, steps: int,
                            n_seeds: int, sample_interval: int,
                            status: StatusWriter) -> dict:
    """Replay the same params with N different seeds; collect end-state
    metrics; report mean ± std. Skips the original seed; uses 0..N."""
    status.update(pass_='C', pct=0.0, msg=f'seed sweep 0/{n_seeds}')
    original_seed = int(get_field(entry, 'seed', 0))
    seeds: list[int] = []
    s = 0
    while len(seeds) < n_seeds:
        if s != original_seed:
            seeds.append(s)
        s += 1
    rows = []
    for i, seed in enumerate(seeds):
        sim = _make_sim(entry['rule'], size=size,
                        params=get_field(entry, 'params', {}) or {},
                        dt=entry.get('dt'), seed=seed,
                        init_variant=entry.get('init_variant'))
        sim._debug_enabled = True
        sim._debug_sample_interval = max(1, int(sample_interval))
        for j in range(steps):
            sim._step_sim()
            if (j % sim._debug_sample_interval) == 0:
                _drain_debug_fence(sim, 0.3)
                _force_debug_sample(sim)
        _force_debug_sample(sim)
        rows.append({'seed': seed, **_end_state_metrics(sim)})
        _release_sim(sim)
        status.update(pct=(i + 1) / len(seeds),
                      msg=f'seed sweep {i + 1}/{len(seeds)}')

    # Aggregate
    summary = {}
    for k in ('active_frac', 'rg', 'mean_c0', 'std_c0'):
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        summary[k] = {'mean': float(vals.mean()),
                      'std':  float(vals.std()),
                      'min':  float(vals.min()),
                      'max':  float(vals.max())}
    # Stability score: low CV on active_frac means seed-robust
    af_mean = summary['active_frac']['mean']
    af_std  = summary['active_frac']['std']
    cv = (af_std / af_mean) if af_mean > 1e-9 else float('inf')
    summary['active_frac_cv'] = float(cv)
    return {'rows': rows, 'summary': summary}


# ── Pass D: parameter sensitivity via Latin-hypercube ──────────────────

def _latin_hypercube(n_rows: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Standard LHS in [0,1]^n_dim with one sample per stratum per dim."""
    u = rng.random((n_rows, n_dim))
    cut = np.arange(n_rows) / n_rows
    out = np.empty_like(u)
    for d in range(n_dim):
        perm = rng.permutation(n_rows)
        out[:, d] = cut[perm] + u[perm, d] / n_rows
    return out


def pass_d_param_sensitivity(entry: dict, *, size: int, steps: int,
                             n_rows: int, span: float,
                             sample_interval: int,
                             status: StatusWriter) -> dict:
    """LHS perturbation around the entry's params. ``span`` is the
    fractional half-width: each param sweeps [p*(1-span), p*(1+span)].
    Params with value 0 sweep ±span absolute. Returns rows + a per-param
    elasticity (Δmetric / Δparam normalised)."""
    params = dict(get_field(entry, 'params', {}) or {})
    if not params:
        return {'rows': [], 'note': 'no params'}
    keys = sorted(params.keys())
    base = np.array([params[k] for k in keys], dtype=np.float64)
    lo = np.where(np.abs(base) > 1e-9, base * (1 - span), -span)
    hi = np.where(np.abs(base) > 1e-9, base * (1 + span),  span)
    rng = np.random.default_rng(0xC0FFEE)
    lhs = _latin_hypercube(n_rows, len(keys), rng)
    samples = lo + lhs * (hi - lo)

    status.update(pass_='D', pct=0.0, msg=f'lhs 0/{n_rows}')
    rows = []
    for i in range(n_rows):
        p = {k: float(samples[i, j]) for j, k in enumerate(keys)}
        sim = _make_sim(entry['rule'], size=size, params=p,
                        dt=entry.get('dt'),
                        seed=int(get_field(entry, 'seed', 0)),
                        init_variant=entry.get('init_variant'))
        sim._debug_enabled = True
        sim._debug_sample_interval = max(1, int(sample_interval))
        for j in range(steps):
            sim._step_sim()
            if (j % sim._debug_sample_interval) == 0:
                _drain_debug_fence(sim, 0.3)
                _force_debug_sample(sim)
        _force_debug_sample(sim)
        rows.append({'params': p, **_end_state_metrics(sim)})
        _release_sim(sim)
        status.update(pct=(i + 1) / n_rows, msg=f'lhs {i + 1}/{n_rows}')

    # Elasticity: Pearson r between each normalised param and active_frac.
    af = np.array([r['active_frac'] for r in rows], dtype=np.float64)
    elasticity = {}
    for j, k in enumerate(keys):
        col = samples[:, j]
        if col.std() < 1e-12 or af.std() < 1e-12:
            elasticity[k] = 0.0
        else:
            elasticity[k] = float(np.corrcoef(col, af)[0, 1])
    return {'rows': rows, 'keys': keys,
            'base': base.tolist(), 'lo': lo.tolist(), 'hi': hi.tolist(),
            'elasticity_active_frac': elasticity}


# ── Pass E: neighbourhood map ──────────────────────────────────────────

def pass_e_neighbours(entry: dict, all_disc: list[dict],
                      parent_index: int, k: int,
                      status: StatusWriter) -> dict:
    """Find k nearest entries among same-rule discoveries by cosine
    distance in normalised parameter space. Each param dim is
    z-scored across the cohort. Returns ranking by distance."""
    status.update(pass_='E', pct=0.0, msg='ranking')
    rule = get_field(entry, 'rule')
    keys = sorted((get_field(entry, 'params', {}) or {}).keys())
    if not keys:
        return {'neighbours': [], 'note': 'no params'}
    cohort_idx = [i for i, d in enumerate(all_disc)
                  if get_field(d, 'rule') == rule and i != parent_index]
    if not cohort_idx:
        return {'neighbours': [], 'note': 'no cohort'}

    def vec(d: dict) -> np.ndarray:
        p = get_field(d, 'params', {}) or {}
        return np.array([float(p.get(k, 0.0)) for k in keys], dtype=np.float64)

    me = vec(entry)
    cohort_vecs = np.stack([vec(all_disc[i]) for i in cohort_idx])
    # z-score per dim using the cohort (includes self for stable scale)
    full = np.vstack([cohort_vecs, me[None, :]])
    mu = full.mean(axis=0)
    sd = full.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    cohort_z = (cohort_vecs - mu) / sd
    me_z     = (me - mu) / sd
    me_norm = np.linalg.norm(me_z) + 1e-12
    cn = np.linalg.norm(cohort_z, axis=1) + 1e-12
    cos = (cohort_z @ me_z) / (cn * me_norm)
    dist = 1.0 - cos
    order = np.argsort(dist)[:k]
    out = []
    for rank, j in enumerate(order):
        idx = cohort_idx[int(j)]
        d = all_disc[idx]
        out.append({
            'rank': rank,
            'index': idx,
            'distance': float(dist[j]),
            'score': float(get_field(d, 'score', 0.0)),
            'params': get_field(d, 'params'),
            'seed':   get_field(d, 'seed'),
            'marked': bool(d.get('marked')),
            'refined': bool(d.get('refinement')),
        })
    status.update(pct=1.0, msg='neighbours done')
    return {'neighbours': out, 'keys': keys}


# ── Driver ─────────────────────────────────────────────────────────────

def refine_one(parent_index: int, all_disc: list[dict], args) -> dict:
    """Run all five passes on a single discovery and return the
    refinement block to be back-written into the parent."""
    entry = all_disc[parent_index]
    h, out_dir = refinement_dir(entry)
    os.makedirs(out_dir, exist_ok=True)
    status = StatusWriter(h)
    t_start = time.time()

    try:
        print(f"[refine] #{parent_index} {get_field(entry, 'rule')} → {h}", flush=True)
        # Pass A — deep replay
        sim, snapshots = pass_a_deep_replay(
            entry, size=args.size, steps=args.steps,
            sample_interval=args.sample_interval,
            status=status, out_dir=out_dir)
        history = list(sim._debug_history)
        end_state = _end_state_metrics(sim)
        _release_sim(sim)

        # Pass B — fingerprint
        fingerprint = pass_b_fingerprint(snapshots, history, status)
        with open(os.path.join(out_dir, 'fingerprint.json'), 'w') as f:
            json.dump(fingerprint, f, indent=2, default=_json_default)

        # Pass C — seed sensitivity
        seed_sweep = pass_c_seed_sensitivity(
            entry, size=args.size, steps=args.steps,
            n_seeds=args.seeds, sample_interval=args.sample_interval,
            status=status)
        with open(os.path.join(out_dir, 'seed_sweep.json'), 'w') as f:
            json.dump(seed_sweep, f, indent=2, default=_json_default)

        # Pass D — parameter sensitivity
        perturbation = pass_d_param_sensitivity(
            entry, size=args.size, steps=args.steps,
            n_rows=args.perturbations, span=args.span,
            sample_interval=args.sample_interval, status=status)
        with open(os.path.join(out_dir, 'perturbation.json'), 'w') as f:
            json.dump(perturbation, f, indent=2, default=_json_default)

        # Pass E — neighbourhood
        neighbours = pass_e_neighbours(entry, all_disc, parent_index,
                                       k=args.neighbours, status=status)
        with open(os.path.join(out_dir, 'neighbours.json'), 'w') as f:
            json.dump(neighbours, f, indent=2, default=_json_default)

        # Verdict heuristic — purely from numbers.
        af_cv = seed_sweep.get('summary', {}).get('active_frac_cv', float('inf'))
        period = fingerprint.get('period', {}) if isinstance(fingerprint.get('period'), dict) else {}
        translation = fingerprint.get('translation', {}) if isinstance(fingerprint.get('translation'), dict) else {}
        growth = fingerprint.get('growth', {}) if isinstance(fingerprint.get('growth'), dict) else {}
        if af_cv < 0.05 and end_state['active_frac'] > 1e-3:
            verdict = 'stable'
        elif period.get('period', 0) and period.get('period', 0) > 1:
            verdict = 'periodic'
        elif translation.get('is_glider'):
            verdict = 'glider'
        elif growth.get('growth_type') == 'exponential':
            verdict = 'expanding'
        elif end_state['active_frac'] < 1e-4:
            verdict = 'extinct'
        elif af_cv > 0.5:
            verdict = 'chaotic'
        else:
            verdict = 'mixed'

        report = {
            'hash': h,
            'parent_index': parent_index,
            'rule': get_field(entry, 'rule'),
            'params': get_field(entry, 'params'),
            'seed': get_field(entry, 'seed'),
            'init_variant': entry.get('init_variant'),
            'dt': entry.get('dt'),
            'config': {
                'size': args.size, 'steps': args.steps,
                'sample_interval': args.sample_interval,
                'seeds': args.seeds, 'perturbations': args.perturbations,
                'span': args.span, 'neighbours': args.neighbours,
            },
            'end_state': end_state,
            'fingerprint': fingerprint,
            'seed_summary': seed_sweep.get('summary'),
            'param_elasticity': perturbation.get('elasticity_active_frac'),
            'neighbours_top': neighbours.get('neighbours', [])[:5],
            'verdict': verdict,
            'wall_seconds': round(time.time() - t_start, 1),
            'completed_at': _iso_now(),
        }
        with open(os.path.join(out_dir, 'report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=_json_default)

        # Back-write the parent entry
        status.update(pass_='writeback', pct=0.5, msg='updating discoveries.json')
        block = {
            'id': h,
            'dir': os.path.relpath(out_dir, THIS_DIR),
            'completed_at': report['completed_at'],
            'verdict': verdict,
            'key_metrics': {
                'active_frac_final': end_state['active_frac'],
                'rg_final': end_state['rg'],
                'active_frac_cv': af_cv,
            },
        }
        _write_back_refinement(parent_index, block)
        status.done(msg=f'verdict={verdict}')
        print(f"[refine]   ✓ {h}  verdict={verdict}  "
              f"({report['wall_seconds']}s)", flush=True)
        return block

    except Exception as e:  # noqa: BLE001  best-effort write
        tb = traceback.format_exc()
        status.fail(msg=str(e))
        print(f"[refine]   ✗ {h} FAILED: {e}\n{tb}", file=sys.stderr, flush=True)
        with open(os.path.join(out_dir, 'error.txt'), 'w') as f:
            f.write(tb)
        return {'id': h, 'error': str(e), 'failed_at': _iso_now()}


# ── CLI ────────────────────────────────────────────────────────────────

def _pick_targets(args, all_disc: list[dict]) -> list[int]:
    if args.idx is not None:
        if not (0 <= args.idx < len(all_disc)):
            sys.exit(f"--idx {args.idx} out of range (0..{len(all_disc) - 1})")
        return [args.idx]
    if args.hash:
        matches = [i for i, d in enumerate(all_disc) if short_hash(d) == args.hash]
        if not matches:
            sys.exit(f"no discovery matches hash {args.hash}")
        return matches
    if args.all_marked:
        marked = [i for i, d in enumerate(all_disc) if d.get('marked')]
        if not marked:
            sys.exit("no marked discoveries found")
        return marked
    sys.exit("specify --idx N, --hash H, or --all-marked")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--idx', type=int, help='Discovery index to refine')
    g.add_argument('--hash', type=str, help='Refine by short hash')
    g.add_argument('--all-marked', action='store_true',
                   help='Refine every discovery with marked=true')

    p.add_argument('--size', type=int, default=96,
                   help='Grid side length for replay (default 96)')
    p.add_argument('--steps', type=int, default=1500,
                   help='Replay step count (default 1500)')
    p.add_argument('--sample-interval', type=int, default=10,
                   help='Debug stats sample period in steps (default 10)')
    p.add_argument('--seeds', type=int, default=8,
                   help='Pass C: alternate seed count (default 8)')
    p.add_argument('--perturbations', type=int, default=20,
                   help='Pass D: LHS rows (default 20)')
    p.add_argument('--span', type=float, default=0.10,
                   help='Pass D: ± fractional perturbation (default 0.10)')
    p.add_argument('--neighbours', type=int, default=10,
                   help='Pass E: k for k-NN (default 10)')
    p.add_argument('--keep-going', action='store_true',
                   help='With --all-marked, do not abort on first failure')
    args = p.parse_args()

    all_disc = _load_discoveries()
    if not all_disc:
        sys.exit(f"no discoveries at {DISC_PATH}")
    targets = _pick_targets(args, all_disc)

    os.makedirs(REF_DIR, exist_ok=True)
    os.makedirs(STATUS_DIR, exist_ok=True)
    print(f"[refine] {len(targets)} target(s) "
          f"size={args.size} steps={args.steps} "
          f"seeds={args.seeds} pert={args.perturbations}", flush=True)

    failures = 0
    for n, parent_index in enumerate(targets):
        # Reload between targets so writebacks accumulate visibly and we
        # always see the latest user edits.
        all_disc = _load_discoveries()
        if not (0 <= parent_index < len(all_disc)):
            print(f"[refine] index {parent_index} no longer valid, skipping",
                  file=sys.stderr)
            continue
        print(f"[refine] === {n + 1}/{len(targets)} ===", flush=True)
        try:
            block = refine_one(parent_index, all_disc, args)
            if 'error' in block:
                failures += 1
                if not args.keep_going and len(targets) > 1:
                    sys.exit(2)
        except KeyboardInterrupt:
            print("[refine] interrupted", file=sys.stderr)
            sys.exit(130)
        except Exception as e:  # noqa: BLE001  per-item failure, skip and continue
            failures += 1
            print(f"[refine] unexpected error on idx={parent_index}: {e}",
                  file=sys.stderr)
            traceback.print_exc()
            if not args.keep_going:
                sys.exit(3)

    print(f"[refine] complete  ok={len(targets) - failures}  failed={failures}",
          flush=True)
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()

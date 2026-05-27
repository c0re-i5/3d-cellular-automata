"""Bug B re-investigation — dual_lenia cross-process determinism.

Per memory, `flagship_dual_lenia` at size=384 was reported to diverge
between independent Python processes at step ≥22 (step 21 identical,
step 22 23M/57M cells differ, max|d|=0.52). In-process determinism
(Probe #1) was 94/94 ok at sizes 32..256 — only multi-process at 384
showed the bug.

This probe spawns N subprocesses, each runs the rule cap steps, hashes
the grid at every checkpoint, and reports the first divergence step.

Suspects to discriminate:
  H1: non-deterministic GLSL (e.g. atomic_add ordering, lockstep race)
      → divergence at all sizes if seeds match exactly, just less obvious
      → falsified if size=256 is rock-stable.
  H2: workgroup-count specific race that only manifests above a size
      threshold (more concurrent groups → more interleavings).
      → divergence appears at size≥some threshold.
  H3: fp32 denormal handling differs run-to-run (driver/PCIe init state)
      → divergence only after field saturates into denormal range.
      → check: are any values < 1e-30 at the divergence step?
  H4: init non-determinism upstream — different CANONICAL_INIT_SIZE
      noise per process due to RNG seeding.
      → falsified by checking step-0 grid hashes identical.

Each subprocess writes (size, step, sha256_of_grid_bytes, min, max,
denorm_count) lines to stdout; the driver collects, groups by step,
and reports the first step with non-unique hash.

Usage::

    python -m ca_debug.bug_b_dual_lenia
    python -m ca_debug.bug_b_dual_lenia --size 256 --cap 30 --trials 4
    python -m ca_debug.bug_b_dual_lenia --size 384 --cap 25 --trials 3 --rule flagship_dual_lenia
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time


_SUBPROC_BANNER = '__BUGB_RESULT__'


def _subprocess_main():
    """Run inside each spawned subprocess. Emits a single JSON line."""
    rule = os.environ['_BUGB_RULE']
    size = int(os.environ['_BUGB_SIZE'])
    seed = int(os.environ['_BUGB_SEED'])
    cap = int(os.environ['_BUGB_CAP'])
    every = int(os.environ['_BUGB_EVERY'])

    # Quiet imports.
    import contextlib
    import io
    import numpy as np
    with contextlib.redirect_stdout(io.StringIO()):
        from test_harness import HeadlessRunner, create_headless_context
    _w, ctx = create_headless_context()

    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed)

    checkpoints = []

    def snapshot(step):
        g = np.asarray(r.read_grid())
        h = hashlib.sha256(g.tobytes()).hexdigest()
        absg = np.abs(g)
        nz = absg[absg > 0]
        denorm = int(np.count_nonzero((nz > 0) & (nz < 1.175494e-38)))
        finite = bool(np.isfinite(g).all())
        checkpoints.append({
            'step': step,
            'hash': h,
            'min': float(g.min()),
            'max': float(g.max()),
            'mean': float(g.mean()),
            'denorm_count': denorm,
            'finite': finite,
        })

    snapshot(0)
    for s in range(1, cap + 1):
        r.step()
        if s % every == 0 or s == cap:
            snapshot(s)
    try: r.release()
    except Exception: pass  # noqa: BLE001

    sys.stdout.write(_SUBPROC_BANNER + json.dumps({'checkpoints': checkpoints}) + '\n')
    sys.stdout.flush()


def _spawn_trial(rule: str, size: int, seed: int, cap: int, every: int,
                 trial_idx: int) -> dict:
    env = dict(os.environ,
               _BUGB_RULE=rule, _BUGB_SIZE=str(size),
               _BUGB_SEED=str(seed), _BUGB_CAP=str(cap),
               _BUGB_EVERY=str(every),
               _BUGB_SUBPROC='1',
               CA_HARNESS_ALLOW_UNDERSIZE='1')
    cmd = [sys.executable, '-m', 'ca_debug.bug_b_dual_lenia', '--_subproc']
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=600)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        return {'trial': trial_idx, 'error': True,
                'returncode': proc.returncode,
                'stderr_tail': proc.stderr.splitlines()[-10:],
                'elapsed_s': elapsed}
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith(_SUBPROC_BANNER):
            payload = json.loads(line[len(_SUBPROC_BANNER):])
            break
    if payload is None:
        return {'trial': trial_idx, 'error': True,
                'reason': 'no banner line',
                'stdout_tail': proc.stdout.splitlines()[-10:],
                'elapsed_s': elapsed}
    return {'trial': trial_idx, 'error': False,
            'elapsed_s': elapsed, **payload}


def _analyse(trials: list[dict]):
    """Find first step where hashes diverge across trials."""
    ok_trials = [t for t in trials if not t.get('error')]
    if len(ok_trials) < 2:
        return {'divergent': None, 'note': 'insufficient successful trials'}

    by_step: dict[int, list[dict]] = {}
    for tr in ok_trials:
        for cp in tr['checkpoints']:
            by_step.setdefault(cp['step'], []).append({
                'trial': tr['trial'], **cp,
            })

    first_divergent = None
    for step in sorted(by_step):
        ckpts = by_step[step]
        hashes = {c['hash'] for c in ckpts}
        if len(hashes) > 1 and first_divergent is None:
            first_divergent = {'step': step, 'checkpoints': ckpts,
                               'n_distinct': len(hashes)}
            break

    last_agree = None
    for step in sorted(by_step):
        hashes = {c['hash'] for c in by_step[step]}
        if len(hashes) == 1:
            last_agree = step

    return {'divergent': first_divergent,
            'last_agree': last_agree,
            'all_steps': sorted(by_step)}


def main(argv=None):
    if '--_subproc' in (argv or sys.argv[1:]):
        _subprocess_main()
        return 0

    ap = argparse.ArgumentParser()
    ap.add_argument('--rule', default='flagship_dual_lenia')
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--seed', type=int, default=1001)
    ap.add_argument('--cap', type=int, default=30)
    ap.add_argument('--every', type=int, default=1,
                    help='Hash every Nth step (default 1).')
    ap.add_argument('--trials', type=int, default=3,
                    help='Number of independent subprocesses (default 3).')
    ap.add_argument('--json', help='Write full report JSON.')
    args = ap.parse_args(argv)

    print(f'Bug B probe — rule={args.rule} size={args.size} seed={args.seed} '
          f'cap={args.cap} trials={args.trials}')
    trials = []
    for i in range(args.trials):
        sys.stdout.write(f'  trial {i+1}/{args.trials}... ')
        sys.stdout.flush()
        tr = _spawn_trial(args.rule, args.size, args.seed,
                          args.cap, args.every, i)
        trials.append(tr)
        if tr.get('error'):
            print(f'ERROR ({tr.get("returncode", "?")}) in {tr["elapsed_s"]:.1f}s')
            for line in tr.get('stderr_tail') or tr.get('stdout_tail') or []:
                print(f'      {line}')
        else:
            ck = tr['checkpoints']
            print(f'ok ({len(ck)} checkpoints, {tr["elapsed_s"]:.1f}s, '
                  f'final hash {ck[-1]["hash"][:12]})')

    analysis = _analyse(trials)
    print()
    if analysis.get('divergent') is None:
        print(f'  DETERMINISTIC — all {sum(1 for t in trials if not t.get("error"))} trials '
              f'agree at every checkpoint through step {analysis.get("last_agree")}')
    else:
        d = analysis['divergent']
        print(f'  DIVERGENT at step {d["step"]} '
              f'({d["n_distinct"]} distinct hashes across trials)')
        print(f'  Last agreement: step {analysis["last_agree"]}')
        for c in d['checkpoints']:
            denorm = c['denorm_count']
            print(f'    trial {c["trial"]}: hash={c["hash"][:16]} '
                  f'min={c["min"]:+.3e} max={c["max"]:+.3e} '
                  f'mean={c["mean"]:+.3e} denorms={denorm} '
                  f'finite={c["finite"]}')
        # Run-to-run summary
        denorm_counts = [c['denorm_count'] for c in d['checkpoints']]
        if max(denorm_counts) > 0:
            print(f'  H3 evidence: denormals present at divergence step '
                  f'(counts {denorm_counts})')
        else:
            print(f'  H3 falsified: no denormals at divergence step')

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'rule': args.rule, 'size': args.size,
                       'seed': args.seed, 'cap': args.cap,
                       'trials': trials, 'analysis': analysis},
                      f, indent=2, default=str)
        print(f'\nwrote {args.json}')

    return 0 if analysis.get('divergent') is None else 1


if __name__ == '__main__':
    sys.exit(main())

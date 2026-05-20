#!/usr/bin/env python3
"""3D Cellular Automata — live dashboard (TUI).

A read-only Textual app that polls:
  - runs/.status/*.json           (per-worker search state, one file per PID)
  - discoveries.json              (most-recent discoveries, mtime-gated)
  - nvidia-smi                    (GPU util/mem/temp/power)
  - psutil                        (CPU util/mem)

Designed for OBS-streamable use:
  - No file paths, no usernames, no command lines surfaced to the UI.
  - Stale data fades rather than disappears (no flicker).
  - Solid block characters for compression-friendly output.

Run:
    python ca_dashboard.py

Quit: q or Ctrl+C
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Static

# Discovery-entry field access: strict on v1+ entries (loud KeyError on
# missing required field), lenient on legacy entries and on dashboard-
# internal dicts like worker status / live-stream payloads.
from schema import get_field

REPO_ROOT = Path(__file__).resolve().parent
STATUS_DIR = REPO_ROOT / 'runs' / '.status'
DISCOVERIES_PATH = REPO_ROOT / 'discoveries.json'

# How long since a status file was last updated before we mark it stale.
STALE_AFTER_S = 30.0

# Polling intervals (seconds). Cheap; Textual schedules these on the event loop.
POLL_GPU_S = 1.0
POLL_CPU_S = 1.0
POLL_STATUS_S = 0.5
POLL_DISCOVERIES_S = 2.0

# How many recent discoveries to show in the panel (it grows with terminal
# height; this is a hard cap on rows).
RECENT_DISCOVERIES_N = 16

# Stream-friendly viewport target. 1920x1080 at ~16-18pt monospace gives
# roughly 160 cols x 45 rows; the layout is fluid so larger terminals work
# fine, but we warn if the user is well below this so streamed text stays
# readable on phone viewers.
TARGET_COLS = 160
TARGET_ROWS = 42

# Max number of live (per-session) discoveries kept in memory. The dashboard
# only ever shows the most recent N rows, but the full history backs the
# leaderboard, per-rule counts, and discoveries/min sparkline.
# Override via env: CA_DASHBOARD_LIVE_CAP=20000
LIVE_DISCOVERY_CAP = int(os.environ.get('CA_DASHBOARD_LIVE_CAP', '50000'))

# A status file whose owning PID no longer exists AND whose mtime is older
# than this is considered abandoned and removed from disk so the dashboard
# stops showing it. Without this, Ctrl+C — which kills bash before its
# children can run their atexit handlers — leaves stale (dead) rows.
DEAD_STATUS_PRUNE_S = 5.0

# When True, hide identifiers (pid, hostname) that don't help viewers.
STREAM_MODE = os.environ.get('CA_DASHBOARD_STREAM', '0') == '1'

# Unicode block ramps for sparkline + bar drawing.
BLOCKS = ' ▁▂▃▄▅▆▇█'
BAR_FILL = '█'
BAR_EMPTY = '░'


# ── Data sources ────────────────────────────────────────────────────────────

def bar(frac: float, width: int = 28) -> str:
    """Solid Unicode bar. Compression-friendly (no per-frame jitter)."""
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return BAR_FILL * filled + BAR_EMPTY * (width - filled)


def sparkline(
    values: list[float],
    width: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> str:
    """Unicode block sparkline.

    - If `width` is given and the series is shorter, left-pad with spaces so
      the sparkline visually grows from the right.
    - If the series is longer than `width`, downsample by mean-of-chunks.
    - `vmin`/`vmax` fix the y-range; otherwise auto-scale to the series.
    """
    if not values:
        return ' ' * (width or 0)
    series = list(values)
    if width and len(series) > width:
        # Downsample by averaging contiguous chunks.
        chunk = len(series) / width
        out_series: list[float] = []
        for i in range(width):
            lo = int(i * chunk)
            hi = max(int((i + 1) * chunk), lo + 1)
            slab = series[lo:hi]
            out_series.append(sum(slab) / len(slab))
        series = out_series
    if vmin is None:
        vmin = min(series)
    if vmax is None:
        vmax = max(series)
    rng = max(vmax - vmin, 1e-9)
    n = len(BLOCKS) - 1
    s = ''.join(
        BLOCKS[max(0, min(n, int(round((v - vmin) / rng * n))))]
        for v in series
    )
    if width and len(series) < width:
        s = ' ' * (width - len(series)) + s
    return s


class History:
    """Bounded ring of (timestamp, value) samples."""

    __slots__ = ('maxlen', 'values', 'times')

    def __init__(self, maxlen: int = 60) -> None:
        self.maxlen = maxlen
        self.values: list[float] = []
        self.times: list[float] = []

    def push(self, value: float, t: float | None = None) -> None:
        self.values.append(float(value))
        self.times.append(t if t is not None else time.time())
        if len(self.values) > self.maxlen:
            self.values = self.values[-self.maxlen:]
            self.times = self.times[-self.maxlen:]

    def spark(
        self,
        width: int | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> str:
        return sparkline(self.values, width=width, vmin=vmin, vmax=vmax)

    def rate_per_minute(self, window_s: float = 60.0) -> float:
        """Approximate growth rate of `value` over the last `window_s`."""
        if len(self.values) < 2:
            return 0.0
        now = self.times[-1]
        # Find the oldest sample within the window.
        i = 0
        for i in range(len(self.times) - 1, -1, -1):
            if now - self.times[i] >= window_s:
                break
        dv = self.values[-1] - self.values[i]
        dt = max(now - self.times[i], 1e-9)
        return dv * 60.0 / dt


def fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f'{int(seconds)}s'
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f'{m}m {s:02d}s'
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f'{h}h {m:02d}m'


def fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f'{int(seconds)}s ago'
    if seconds < 3600:
        return f'{int(seconds // 60)}m ago'
    if seconds < 86400:
        return f'{int(seconds // 3600)}h ago'
    return f'{int(seconds // 86400)}d ago'


@dataclass
class GPUSample:
    name: str = '—'
    util: float = 0.0          # 0..1
    mem_used_gb: float = 0.0
    mem_total_gb: float = 0.0
    temp_c: float = 0.0
    power_w: float = 0.0
    power_max_w: float = 0.0
    fan_pct: float = 0.0
    ok: bool = False


def read_gpu() -> GPUSample:
    """Snapshot via nvidia-smi --query-gpu=... --format=csv,noheader,nounits.

    Returns an empty sample if nvidia-smi is missing or fails.
    """
    try:
        out = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=name,utilization.gpu,memory.used,memory.total,'
                'temperature.gpu,power.draw,power.limit,fan.speed',
                '--format=csv,noheader,nounits',
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip().splitlines()
    except (FileNotFoundError, subprocess.SubprocessError):
        return GPUSample()
    if not out:
        return GPUSample()
    parts = [p.strip() for p in out[0].split(',')]
    if len(parts) < 8:
        return GPUSample()
    try:
        return GPUSample(
            name=parts[0],
            util=float(parts[1]) / 100.0,
            mem_used_gb=float(parts[2]) / 1024.0,
            mem_total_gb=float(parts[3]) / 1024.0,
            temp_c=float(parts[4]),
            power_w=float(parts[5]),
            power_max_w=float(parts[6]),
            fan_pct=float(parts[7]) if parts[7] not in ('', '[N/A]') else 0.0,
            ok=True,
        )
    except ValueError:
        return GPUSample()


@dataclass
class CPUSample:
    util: float = 0.0           # 0..1, mean across logical cores
    mem_used_gb: float = 0.0
    mem_total_gb: float = 0.0


def read_cpu() -> CPUSample:
    util = psutil.cpu_percent(interval=None) / 100.0  # non-blocking
    vm = psutil.virtual_memory()
    return CPUSample(
        util=util,
        mem_used_gb=vm.used / (1024 ** 3),
        mem_total_gb=vm.total / (1024 ** 3),
    )


def read_status_files() -> list[dict[str, Any]]:
    """Return a list of per-worker status dicts, sorted by start time.

    Side effect: status files whose owning PID no longer exists and whose
    mtime is older than ``DEAD_STATUS_PRUNE_S`` are deleted on the spot.
    This is what cleans up after a Ctrl+C’d batch — bash kills its python
    children before their atexit handlers can run, leaving the file behind.
    """
    if not STATUS_DIR.exists():
        return []
    out: list[dict[str, Any]] = []
    now = time.time()
    for p in STATUS_DIR.glob('*.json'):
        try:
            st = p.stat()
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # File-mtime fallback if the writer somehow didn't set updated_at.
        if 'updated_at' not in data:
            data['updated_at'] = st.st_mtime

        # Prune abandoned files: PID gone + file is no longer being touched.
        pid = int(data.get('pid', 0) or 0)
        age = now - float(data.get('updated_at', 0))
        if pid and age > DEAD_STATUS_PRUNE_S and not psutil.pid_exists(pid):
            try:
                p.unlink()
            except OSError:
                pass
            continue

        out.append(data)
    out.sort(key=lambda d: d.get('started_at', 0))
    return out


# ── Discoveries tail (mtime-gated; we only re-read if the file changed) ────

class DiscoveriesTail:
    """Cheap tail-of-array reader for the JSON discoveries file.

    Loading 21k+ entries every poll would be wasteful. We re-parse only when
    mtime changes, and keep just the last N entries plus a count.
    """

    def __init__(self, path: Path, keep_last: int = RECENT_DISCOVERIES_N):
        self.path = path
        self.keep_last = keep_last
        self.last_mtime: float = -1.0
        self.total_count: int = 0
        self.recent: list[dict[str, Any]] = []

    def poll(self) -> bool:
        """Re-parse if mtime changed. Returns True if data was refreshed."""
        try:
            st = self.path.stat()
        except OSError:
            return False
        if st.st_mtime == self.last_mtime:
            return False
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(data, list):
            return False
        self.last_mtime = st.st_mtime
        self.total_count = len(data)
        self.recent = data[-self.keep_last:]
        return True


# ── Widgets ─────────────────────────────────────────────────────────────────

class HeaderBar(Static):
    """Top bar: title + clock + uptime."""

    started_at = reactive(time.time())
    tick: reactive[int] = reactive(0)

    def render(self) -> str:
        # `tick` is incremented by the App once per second so this re-renders.
        _ = self.tick
        now = time.time()
        clock = time.strftime('%H:%M:%S', time.gmtime(now))
        up = fmt_dur(now - self.started_at)
        return (
            f'[bold cyan]3D Cellular Automata[/]  ·  '
            f'[dim]live dashboard[/]   '
            f'[bold]{clock} UTC[/]   '
            f'[dim]up {up}[/]'
        )


class GPUPanel(Static):
    sample: reactive[GPUSample] = reactive(GPUSample())
    util_hist: reactive[list[float]] = reactive([])
    pwr_hist: reactive[list[float]] = reactive([])
    temp_hist: reactive[list[float]] = reactive([])

    def render(self) -> str:
        s = self.sample
        if not s.ok:
            return '[bold]GPU[/]  [red]nvidia-smi unavailable[/]'
        mem_frac = s.mem_used_gb / s.mem_total_gb if s.mem_total_gb else 0.0
        pwr_frac = s.power_w / s.power_max_w if s.power_max_w else 0.0
        temp_color = (
            'green' if s.temp_c < 70 else
            'yellow' if s.temp_c < 82 else 'red'
        )
        spark_w = 22
        util_spark = sparkline(self.util_hist, width=spark_w, vmin=0.0, vmax=1.0)
        pwr_spark = sparkline(
            self.pwr_hist, width=spark_w,
            vmin=0.0, vmax=max(s.power_max_w, 1.0),
        )
        # Temp sparkline auto-scaled around recent values, but anchored to a
        # safe floor so a flat-cold GPU doesn't visually amplify noise.
        if self.temp_hist:
            t_lo = min(min(self.temp_hist), s.temp_c) - 2
            t_hi = max(max(self.temp_hist), s.temp_c) + 2
        else:
            t_lo, t_hi = s.temp_c - 2, s.temp_c + 2
        temp_spark = sparkline(self.temp_hist, width=spark_w, vmin=t_lo, vmax=t_hi)
        return (
            f'[bold]GPU[/]  [dim]{s.name}[/]\n'
            f'  util  [cyan]{bar(s.util)}[/]  {s.util * 100:5.1f}%  '
            f'[cyan dim]{util_spark}[/]\n'
            f'  mem   [magenta]{bar(mem_frac)}[/]  '
            f'{s.mem_used_gb:5.2f} / {s.mem_total_gb:5.2f} GB\n'
            f'  pwr   [yellow]{bar(pwr_frac)}[/]  '
            f'{s.power_w:5.0f} / {s.power_max_w:.0f} W  '
            f'[yellow dim]{pwr_spark}[/]\n'
            f'  temp  [{temp_color}]{s.temp_c:5.1f} °C[/]   '
            f'fan {s.fan_pct:3.0f}%  '
            f'[{temp_color} dim]{temp_spark}[/]'
        )


class CPUPanel(Static):
    sample: reactive[CPUSample] = reactive(CPUSample())
    util_hist: reactive[list[float]] = reactive([])

    def render(self) -> str:
        s = self.sample
        mem_frac = s.mem_used_gb / s.mem_total_gb if s.mem_total_gb else 0.0
        spark_w = 22
        util_spark = sparkline(self.util_hist, width=spark_w, vmin=0.0, vmax=1.0)
        return (
            f'[bold]CPU[/]\n'
            f'  util  [cyan]{bar(s.util)}[/]  {s.util * 100:5.1f}%  '
            f'[cyan dim]{util_spark}[/]\n'
            f'  mem   [magenta]{bar(mem_frac)}[/]  '
            f'{s.mem_used_gb:5.1f} / {s.mem_total_gb:5.1f} GB'
        )


class SearchOverviewPanel(Static):
    """Aggregate view across all active workers + discovery throughput.

    Tracks:
      - total trials done / target across workers
      - throughput (trials/min, rolling)
      - ETA to completion of all current workers' targets
      - session best score (highest best_score seen this dashboard session)
      - discoveries/min sparkline (60s buckets)
    """

    trials_done: reactive[int] = reactive(0)
    trials_total: reactive[int] = reactive(0)
    elites_done: reactive[int] = reactive(0)
    elites_total: reactive[int] = reactive(0)
    workers_n: reactive[int] = reactive(0)
    throughput: reactive[float] = reactive(0.0)        # trials per minute
    eta_seconds: reactive[float] = reactive(0.0)
    session_best: reactive[float] = reactive(0.0)
    session_best_rule: reactive[str] = reactive('—')
    disc_per_min: reactive[float] = reactive(0.0)
    disc_spark_values: reactive[list[float]] = reactive([])
    trials_spark_values: reactive[list[float]] = reactive([])
    leaderboard: reactive[list[dict[str, Any]]] = reactive([])  # top-3 [(score,rule,seed)]
    last_disc_age_s: reactive[float] = reactive(0.0)
    live_disc_total: reactive[int] = reactive(0)

    def render(self) -> str:
        spark_w = 32
        if self.trials_total > 0:
            frac = self.trials_done / self.trials_total
        else:
            frac = 0.0
        if self.elites_total > 0:
            elite_frac = self.elites_done / self.elites_total
        else:
            elite_frac = 0.0
        if self.eta_seconds > 0 and self.eta_seconds < 86400 * 7:
            eta_str = fmt_dur(self.eta_seconds)
        elif self.workers_n == 0:
            eta_str = '—'
        else:
            eta_str = '∞'

        disc_spark = sparkline(
            self.disc_spark_values, width=spark_w, vmin=0.0,
        )
        trials_spark = sparkline(
            self.trials_spark_values, width=spark_w, vmin=0.0,
        )

        # Leaderboard line: top-3 session bests, separated by dots.
        if self.leaderboard:
            tops = []
            for i, e in enumerate(self.leaderboard[:3]):
                medal = ('[bright_yellow]🥇[/]', '[white]🥈[/]', '[yellow]🥉[/]')[i]
                tops.append(
                    f"{medal} [bold]{get_field(e, 'score', 0):.3f}[/] "
                    f"[cyan]{get_field(e, 'rule', '?')}[/]"
                )
            leaderboard_line = '  ' + '   '.join(tops)
        else:
            leaderboard_line = '  [dim]top 3 \u2014 (none yet)[/]'

        # Time since last discovery.
        if self.last_disc_age_s <= 0 or self.live_disc_total == 0:
            last_str = '\u2014'
        elif self.last_disc_age_s < 1:
            last_str = '[green]just now[/]'
        else:
            last_str = fmt_age(self.last_disc_age_s)

        return (
            f'[bold]Search Overview[/]  '
            f'[dim]({self.workers_n} worker{"s" if self.workers_n != 1 else ""}, '
            f'{self.live_disc_total} live discoveries, last: {last_str})[/]\n'
            f'  trials   [green]{bar(frac, 32)}[/]  '
            f'{self.trials_done:>6d}/{self.trials_total:<6d}  '
            f'[dim]{self.throughput:5.1f}/min  ETA {eta_str}[/]\n'
            f'  elites   [magenta]{bar(elite_frac, 32)}[/]  '
            f'{self.elites_done:>6d}/{self.elites_total:<6d}\n'
            f'  trials/s [cyan dim]{trials_spark}[/]  '
            f'[dim](last {spark_w * 2}s)[/]\n'
            f'  discov.  [yellow dim]{disc_spark}[/]  '
            f'{self.disc_per_min:4.1f}/min\n'
            f'{leaderboard_line}'
        )


class WorkersPanel(Static):
    rows: reactive[list[dict[str, Any]]] = reactive([])
    # pid -> best-score history (set by App)
    best_history: dict[int, History] = {}

    def render(self) -> str:
        if not self.rows:
            return '[bold]Search Workers[/]\n  [dim]idle — no active search[/]'
        now = time.time()
        # Compact (one line per worker) when there are many workers, so an
        # 8-job batch doesn't get clipped on a streaming-target terminal.
        compact = len(self.rows) > 5
        lines = [
            f'[bold]Search Workers[/]  [dim]({len(self.rows)} active'
            f'{", compact" if compact else ""})[/]'
        ]
        for w in self.rows:
            updated = w.get('updated_at', 0)
            age = now - updated
            stale = age > STALE_AFTER_S
            dead = age > STALE_AFTER_S * 4
            trial = w.get('trial', 0)
            total = w.get('trials_total', 1)
            frac = trial / total if total else 0.0
            elites = w.get('elites_filled', 0)
            elites_max = w.get('elites_max', 1)
            best = w.get('best_score', 0.0)
            phase = w.get('phase', '?')
            rule = w.get('rule', '?')
            metric = w.get('metric', '?')
            size = w.get('size', '?')
            pid = w.get('pid', 0)

            if dead:
                row_color = 'red'
                state_tag = '[red](dead)[/]'
            elif stale:
                row_color = 'dim'
                state_tag = '[red](stale)[/]'
            else:
                row_color = 'white'
                state_tag = ''
            phase_color = 'yellow' if phase == 'boot' else 'cyan'
            if compact:
                # One-line dense format. 12-wide bar, no sparkline.
                head = (
                    f'  [{row_color}][cyan]{rule:<22}[/] '
                    f'[{phase_color}]{phase[:4]:<4}[/] '
                    f'[green]{bar(frac, 12)}[/] '
                    f'{trial:>5d}/{total:<5d} '
                    f'el {elites:3d}/{elites_max:<3d} '
                    f'best [bold]{best:.3f}[/]'
                )
                if state_tag:
                    head += ' ' + state_tag
                head += '[/]'
                lines.append(head)
            else:
                if STREAM_MODE:
                    meta = f'[dim]{metric}, {size}³[/]'
                else:
                    meta = f'[dim]{metric}, size={size}³, pid {pid}[/]'
                head = (
                    f'  [{row_color}][bold]{rule}[/]  {meta}  '
                    f'[{phase_color}]{phase}[/]'
                )
                if state_tag:
                    head += '  ' + state_tag
                head += '[/]'
                hist = self.best_history.get(pid)
                best_spark = hist.spark(width=20, vmin=0.0, vmax=1.0) if hist else ''
                prog = (
                    f'    [green]{bar(frac, 28)}[/]  '
                    f'{trial:>5d}/{total:<5d}  '
                    f'elites {elites:3d}/{elites_max:<3d}  '
                    f'best [bold]{best:.3f}[/]  '
                    f'[green dim]{best_spark}[/]'
                )
                lines.append(head)
                lines.append(prog)
        return '\n'.join(lines)


class DiscoveriesPanel(Static):
    tail: reactive[list[dict[str, Any]]] = reactive([])
    total_count: reactive[int] = reactive(0)
    live_recent: reactive[list[dict[str, Any]]] = reactive([])
    live_total: reactive[int] = reactive(0)

    def render(self) -> str:
        # Prefer the live merged feed (per-worker recent_top) when we have any;
        # otherwise fall back to the on-disk discoveries.json tail.
        use_live = bool(self.live_recent)
        if use_live:
            entries = self.live_recent
            header = (
                f'[bold]Recent Discoveries[/]  '
                f'[dim](live: {self.live_total} this session  ·  '
                f'on-disk: {self.total_count:,})[/]'
            )
        else:
            entries = self.tail
            header = (
                f'[bold]Recent Discoveries[/]  '
                f'[dim](total: {self.total_count:,})[/]'
            )
        lines = [header]
        if not entries:
            lines.append('  [dim]none yet[/]')
            return '\n'.join(lines)
        # Newest first.
        for d in reversed(entries):
            score = get_field(d, 'score', 0.0)
            rule = get_field(d, 'rule', '?')
            star = '★' if score >= 0.7 else '·'
            star_color = (
                'bright_yellow' if score >= 0.85 else
                'yellow' if score >= 0.7 else 'dim'
            )
            lines.append(
                f'  [{star_color}]{star}[/]  '
                f'[bold]{score:5.3f}[/]  '
                f'[cyan]{rule:<24}[/]  '
                f'[dim]seed={get_field(d, "seed", "?")}[/]'
            )
        return '\n'.join(lines)


class RuleStatsPanel(Static):
    """Per-rule live stats: counts, best score, time since last seen.

    Built from the cumulative live discovery feed so it survives worker
    deaths (unlike WorkersPanel which only shows currently-running PIDs).
    """

    rows: reactive[list[dict[str, Any]]] = reactive([])

    def render(self) -> str:
        if not self.rows:
            return (
                '[bold]Per-Rule Stats[/]  [dim](this session)[/]\n'
                '  [dim]no discoveries yet \u2014 waiting on workers[/]'
            )
        now = time.time()
        lines = [
            f'[bold]Per-Rule Stats[/]  [dim](this session, '
            f'{len(self.rows)} rule{"s" if len(self.rows) != 1 else ""})[/]'
        ]
        # rows are pre-sorted by best score, descending.
        for r in self.rows:
            rule = r.get('rule', '?')
            count = r.get('count', 0)
            best = r.get('best', 0.0)
            last_t = r.get('last_t', 0.0)
            age = now - last_t if last_t else 1e9
            star_color = (
                'bright_yellow' if best >= 0.85 else
                'yellow' if best >= 0.7 else
                'green' if best >= 0.5 else 'dim'
            )
            age_str = fmt_age(age) if age < 86400 else '\u2014'
            lines.append(
                f'  [{star_color}]\u2605[/]  '
                f'[cyan]{rule:<24}[/]  '
                f'count [bold]{count:>4d}[/]  '
                f'best [bold]{best:.3f}[/]  '
                f'[dim]last {age_str}[/]'
            )
        return '\n'.join(lines)


class FooterBar(Static):
    """Status of underlying data sources."""

    gpu_ok = reactive(True)
    workers_n = reactive(0)
    discoveries_n = reactive(0)

    def render(self) -> str:
        gpu_tag = '[green]●[/] gpu' if self.gpu_ok else '[red]●[/] gpu'
        return (
            f'[dim]q to quit · {gpu_tag} · '
            f'workers {self.workers_n} · '
            f'discoveries {self.discoveries_n:,}[/]'
        )


class LivePreviewsPanel(Static):
    """Mosaic of tiny live thumbnails — one per active worker.

    Each worker writes an RGB preview (base64) into its status JSON
    after every trial — typically a 40×40 axonometric voxel render
    (`mode='iso'`) or a 32×32 max-projection (`mode='maxproj'`). We
    render them as 24-bit-color Unicode half-blocks (▀ = top RGB on
    bottom RGB), so an HxW preview becomes W cols × ceil(H/2) rows of
    text per thumbnail. Thumbnail size is read per-preview from the
    status JSON, so iso and maxproj workers can coexist in the same
    mosaic without layout glitches. The mosaic lays out responsively
    based on terminal columns.
    """

    workers: reactive[list[dict[str, Any]]] = reactive([])

    # 8-stop inferno fallback, used when a worker hasn't reported a
    # preview yet (or the JSON is malformed). Renders as a flat dim block
    # so the layout doesn't reflow when workers come online.
    _PLACEHOLDER_RGB = (32, 12, 48)
    # Default thumbnail dimensions when no worker has reported a preview
    # yet (used purely for layout math — the placeholder block is solid).
    _DEFAULT_PREVIEW_W = 40
    _DEFAULT_PREVIEW_H = 40

    def render(self) -> str:
        import base64
        if not self.workers:
            return '[bold]Live Previews[/]  [dim]no workers reporting[/]'

        # Decode previews up front so we can iterate row-major.
        # We also need each preview's pixel size to lay out the mosaic,
        # so the per-row arithmetic adapts to whatever the workers send
        # (iso → 40, maxproj → 32, future modes → whatever).
        decoded: list[dict[str, Any]] = []
        now = time.time()
        # Cap at 8 thumbnails (2 visual rows of 4 typical) to keep the
        # panel on-screen when the search fans out wider than that.
        for w in self.workers[:8]:
            preview = w.get('preview')
            rgb = None
            pw, ph = self._DEFAULT_PREVIEW_W, self._DEFAULT_PREVIEW_H
            if preview and isinstance(preview, dict):
                try:
                    raw = base64.b64decode(preview.get('b64', ''))
                    h = int(preview.get('h', self._DEFAULT_PREVIEW_H))
                    bw = int(preview.get('w', self._DEFAULT_PREVIEW_W))
                    if len(raw) == h * bw * 3:
                        rgb = (raw, h, bw)
                        pw, ph = bw, h
                except Exception:  # noqa: BLE001  malformed preview, skip
                    rgb = None
            age = now - float(w.get('updated_at', 0))
            decoded.append({
                'rule':   w.get('rule', '?'),
                'best':   float(w.get('best_score', 0.0)),
                'rgb':    rgb,
                'pw':     pw,
                'ph':     ph,
                'stale':  age > 10.0,
            })

        # Layout: each thumbnail is `pw` chars wide + `gap` char gap.
        # All thumbs in a given row share the same width (use the max
        # so mixed iso/maxproj mosaics still line up cleanly).
        gap = 2
        thumb_w = max(d['pw'] for d in decoded)
        thumb_h_chars = (max(d['ph'] for d in decoded) + 1) // 2
        try:
            cols = max(40, self.app.size.width)
        except Exception:  # noqa: BLE001  terminal size probe, use default
            cols = 160
        per_row = max(1, (cols - 4) // (thumb_w + gap))
        per_row = min(per_row, len(decoded), 8)

        out_lines: list[str] = ['[bold]Live Previews[/]']
        for row_start in range(0, len(decoded), per_row):
            row = decoded[row_start:row_start + per_row]

            # Caption line: rule name + best score (truncated to thumb_w).
            caps = []
            for w in row:
                rule = w['rule']
                if len(rule) > thumb_w - 8:
                    rule = rule[:thumb_w - 9] + '…'
                tag = 'dim' if w['stale'] else 'bold cyan'
                caps.append(f'[{tag}]{rule:<{thumb_w - 7}}[/] [bold]{w["best"]:.3f}[/]')
            out_lines.append((' ' * gap).join(caps))

            # Half-block art: each text row encodes two pixel rows.
            for cy in range(thumb_h_chars):
                y_top = cy * 2
                y_bot = y_top + 1
                segs = []
                for w in row:
                    rgb = w['rgb']
                    if rgb is None:
                        # Placeholder: flat dim block.
                        r, g, b = self._PLACEHOLDER_RGB
                        seg = (
                            f'[rgb({r},{g},{b}) on rgb({r},{g},{b})]'
                            + ('▀' * thumb_w) + '[/]'
                        )
                    else:
                        raw, h, bw = rgb
                        chunks = []
                        for x in range(thumb_w):
                            if x < bw and y_top < h:
                                o_top = (y_top * bw + x) * 3
                                tr, tg, tb = (raw[o_top], raw[o_top + 1],
                                              raw[o_top + 2])
                            else:
                                tr = tg = tb = 0
                            if x < bw and y_bot < h:
                                o_bot = (y_bot * bw + x) * 3
                                br, bg, bb = (raw[o_bot], raw[o_bot + 1],
                                              raw[o_bot + 2])
                            else:
                                br = bg = bb = 0
                            chunks.append(
                                f'[rgb({tr},{tg},{tb}) on rgb({br},{bg},{bb})]▀[/]'
                            )
                        seg = ''.join(chunks)
                    segs.append(seg)
                out_lines.append((' ' * gap).join(segs))

        return '\n'.join(out_lines)


# ── App ─────────────────────────────────────────────────────────────────────

class CADashboard(App):
    CSS = """
    Screen { background: $surface; }
    HeaderBar { height: 1; padding: 0 1; background: $boost; color: $text; }
    FooterBar { dock: bottom; height: 1; padding: 0 1; background: $boost; }
    .panel { border: round $primary; padding: 0 1; margin: 0; overflow-y: auto; }
    GPUPanel { width: 1fr; height: 8; }
    CPUPanel { width: 1fr; height: 8; }
    SearchOverviewPanel { height: 10; }
    .midrow { height: 2fr; min-height: 12; }
    WorkersPanel { width: 2fr; }
    RuleStatsPanel { width: 1fr; }
    DiscoveriesPanel { height: 1fr; min-height: 10; }
    LivePreviewsPanel { height: auto; min-height: 23; }
    """

    BINDINGS = [
        ('q', 'quit', 'Quit'),
        ('ctrl+c', 'quit', 'Quit'),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.discoveries = DiscoveriesTail(DISCOVERIES_PATH)
        # Histories driving sparklines.
        self.gpu_util_hist = History(maxlen=120)
        self.gpu_pwr_hist = History(maxlen=120)
        self.gpu_temp_hist = History(maxlen=120)
        self.cpu_util_hist = History(maxlen=120)
        # Per-pid best-score history (shared with WorkersPanel.best_history).
        self.worker_best: dict[int, History] = {}
        # Per-pid trial history for aggregate throughput.
        self.worker_trials: dict[int, History] = {}
        # Discovery-count history for discoveries/min sparkline.
        self.disc_hist = History(maxlen=600)
        self.session_best: float = 0.0
        self.session_best_rule: str = '—'
        # Cumulative live discovery feed merged across workers, keyed by
        # (pid, trial) so re-reads of the same status file don't duplicate.
        self.live_disc_seen: set[tuple[int, int]] = set()
        self.live_disc: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield HeaderBar(id='header')
        with Vertical():
            with Horizontal():
                yield GPUPanel(classes='panel', id='gpu')
                yield CPUPanel(classes='panel', id='cpu')
            yield SearchOverviewPanel(classes='panel', id='overview')
            with Horizontal(classes='midrow'):
                yield WorkersPanel(classes='panel', id='search_workers')
                yield RuleStatsPanel(classes='panel', id='rule_stats')
            yield LivePreviewsPanel(classes='panel', id='previews')
            yield DiscoveriesPanel(classes='panel', id='discoveries')
        yield FooterBar(id='footer')

    def on_mount(self) -> None:
        # Prime psutil's cpu_percent (first call returns 0.0).
        psutil.cpu_percent(interval=None)
        self.set_interval(POLL_GPU_S, self._tick_gpu)
        self.set_interval(POLL_CPU_S, self._tick_cpu)
        self.set_interval(POLL_STATUS_S, self._tick_status)
        self.set_interval(POLL_DISCOVERIES_S, self._tick_discoveries)
        self.set_interval(1.0, self._tick_clock)
        # Run each tick once immediately so the UI isn't blank for a second.
        self._tick_gpu()
        self._tick_cpu()
        self._tick_status()
        self._tick_discoveries()

    # ── tick handlers ───────────────────────────────────────────

    def _tick_clock(self) -> None:
        # Bump a reactive on the header so its render() runs every second.
        h = self.query_one('#header', HeaderBar)
        h.tick = h.tick + 1

    def _tick_gpu(self) -> None:
        s = read_gpu()
        panel = self.query_one('#gpu', GPUPanel)
        panel.sample = s
        if s.ok:
            self.gpu_util_hist.push(s.util)
            self.gpu_pwr_hist.push(s.power_w)
            self.gpu_temp_hist.push(s.temp_c)
            panel.util_hist = list(self.gpu_util_hist.values)
            panel.pwr_hist = list(self.gpu_pwr_hist.values)
            panel.temp_hist = list(self.gpu_temp_hist.values)
        self.query_one('#footer', FooterBar).gpu_ok = s.ok

    def _tick_cpu(self) -> None:
        s = read_cpu()
        panel = self.query_one('#cpu', CPUPanel)
        panel.sample = s
        self.cpu_util_hist.push(s.util)
        panel.util_hist = list(self.cpu_util_hist.values)

    def _tick_status(self) -> None:
        workers = read_status_files()
        now = time.time()
        # Per-worker best & trial histories, keyed by pid.
        live_pids: set[int] = set()
        for w in workers:
            pid = w.get('pid', 0)
            if not pid:
                continue
            live_pids.add(pid)
            best = float(w.get('best_score', 0.0))
            self.worker_best.setdefault(pid, History(maxlen=120)).push(best, now)
            self.worker_trials.setdefault(pid, History(maxlen=120)).push(
                float(w.get('trial', 0)), now,
            )
            if best > self.session_best:
                self.session_best = best
                self.session_best_rule = w.get('rule', '?')
        # GC dead pids that haven't reported in a while.
        for pid in list(self.worker_best.keys()):
            if pid in live_pids:
                continue
            hist = self.worker_best[pid]
            if not hist.times or now - hist.times[-1] > 300:
                self.worker_best.pop(pid, None)
                self.worker_trials.pop(pid, None)

        wpanel = self.query_one('#search_workers', WorkersPanel)
        wpanel.best_history = self.worker_best
        wpanel.rows = workers

        # Push the same per-worker rows (with their preview blobs) into
        # the previews panel. Sort by best_score so the leaders show
        # in the top-left of the mosaic.
        ppanel = self.query_one('#previews', LivePreviewsPanel)
        ppanel.workers = sorted(
            workers, key=lambda w: float(w.get('best_score', 0.0)), reverse=True,
        )

        # Merge per-worker recent_top into the live discoveries feed.
        for w in workers:
            pid = w.get('pid', 0)
            if not pid:
                continue
            for entry in w.get('recent_top') or []:
                key = (pid, int(entry.get('trial', 0)))
                if key in self.live_disc_seen:
                    continue
                self.live_disc_seen.add(key)
                # Stamp dedupe key into the entry so we can rebuild the
                # seen-set after pruning the cap.
                entry = dict(entry)
                entry['pid_key'] = pid
                self.live_disc.append(entry)
        # Cap memory: keep the last LIVE_DISCOVERY_CAP live discoveries.
        if len(self.live_disc) > LIVE_DISCOVERY_CAP:
            # Drop the oldest entries; rebuild the dedupe set to match.
            self.live_disc = self.live_disc[-LIVE_DISCOVERY_CAP:]
            self.live_disc_seen = {
                (e.get('pid_key', 0), int(e.get('trial', 0)))
                for e in self.live_disc
            }

        dpanel = self.query_one('#discoveries', DiscoveriesPanel)
        # Show the most recent N (newest-first ordering happens in render()).
        dpanel.live_recent = self.live_disc[-RECENT_DISCOVERIES_N:]
        dpanel.live_total = len(self.live_disc)

        # Per-rule aggregates from the cumulative live feed.
        per_rule: dict[str, dict[str, Any]] = {}
        for e in self.live_disc:
            rule = get_field(e, 'rule', '?')
            score = float(get_field(e, 'score', 0.0))
            # 't' is a dashboard-internal timestamp on live-stream entries,
            # not a v1 discovery field — keep plain .get().
            t_ = float(e.get('t', 0.0))
            slot = per_rule.setdefault(
                rule, {'rule': rule, 'count': 0, 'best': 0.0, 'last_t': 0.0}
            )
            slot['count'] += 1
            if score > slot['best']:
                slot['best'] = score
            if t_ > slot['last_t']:
                slot['last_t'] = t_
        rule_rows = sorted(
            per_rule.values(), key=lambda r: r['best'], reverse=True,
        )
        self.query_one('#rule_stats', RuleStatsPanel).rows = rule_rows

        # Leaderboard: top-3 unique-rule bests across the live feed.
        seen_rules: set[str] = set()
        leaderboard: list[dict[str, Any]] = []
        for e in sorted(
            self.live_disc, key=lambda x: float(get_field(x, 'score', 0)), reverse=True,
        ):
            r = get_field(e, 'rule', '?')
            if r in seen_rules:
                continue
            seen_rules.add(r)
            leaderboard.append(e)
            if len(leaderboard) >= 3:
                break
        opanel_lb = self.query_one('#overview', SearchOverviewPanel)
        opanel_lb.leaderboard = leaderboard
        opanel_lb.live_disc_total = len(self.live_disc)
        if self.live_disc:
            last_t = float(self.live_disc[-1].get('t', 0.0))
            opanel_lb.last_disc_age_s = max(0.0, now - last_t)
        else:
            opanel_lb.last_disc_age_s = 0.0

        # Aggregate stats.
        trials_done = sum(w.get('trial', 0) for w in workers)
        trials_total = sum(w.get('trials_total', 0) for w in workers)
        elites_done = sum(w.get('elites_filled', 0) for w in workers)
        elites_total = sum(w.get('elites_max', 0) for w in workers)
        # Throughput: sum across workers of recent trials/min.
        throughput = 0.0
        for pid in live_pids:
            h = self.worker_trials.get(pid)
            if h is not None:
                throughput += h.rate_per_minute(window_s=30.0)
        eta_s = 0.0
        if throughput > 0 and trials_total > trials_done:
            eta_s = (trials_total - trials_done) / (throughput / 60.0)

        # Trials-per-second sparkline derived from the cumulative aggregate.
        # Push aggregate trials_done into a dashboard-level history so we can
        # spark deltas. We piggyback on disc_hist's pattern: keep a parallel.
        self._push_aggregate_trials(trials_done, now)

        opanel = self.query_one('#overview', SearchOverviewPanel)
        opanel.workers_n = len(workers)
        opanel.trials_done = trials_done
        opanel.trials_total = trials_total
        opanel.elites_done = elites_done
        opanel.elites_total = elites_total
        opanel.throughput = throughput
        opanel.eta_seconds = eta_s
        opanel.session_best = self.session_best
        opanel.session_best_rule = self.session_best_rule
        opanel.trials_spark_values = list(self._trials_delta_series())

        self.query_one('#footer', FooterBar).workers_n = len(workers)

    def _push_aggregate_trials(self, trials_done: int, now: float) -> None:
        if not hasattr(self, '_agg_trials_hist'):
            self._agg_trials_hist = History(maxlen=240)
        self._agg_trials_hist.push(float(trials_done), now)

    def _trials_delta_series(self) -> list[float]:
        """Per-tick deltas of total trials (clamped ≥0). Restart-safe."""
        h = getattr(self, '_agg_trials_hist', None)
        if h is None or len(h.values) < 2:
            return []
        deltas: list[float] = []
        prev = h.values[0]
        for v in h.values[1:]:
            d = v - prev
            deltas.append(d if d >= 0 else 0.0)
            prev = v
        return deltas

    def _tick_discoveries(self) -> None:
        refreshed = self.discoveries.poll()
        now = time.time()
        # Always sample the current count so the sparkline keeps moving.
        self.disc_hist.push(float(self.discoveries.total_count), now)
        if refreshed:
            panel = self.query_one('#discoveries', DiscoveriesPanel)
            panel.tail = list(self.discoveries.recent)
            panel.total_count = self.discoveries.total_count
        # Update overview discovery stats.
        opanel = self.query_one('#overview', SearchOverviewPanel)
        opanel.disc_per_min = self.disc_hist.rate_per_minute(window_s=60.0)
        # Sparkline = per-tick deltas (so it shows discovery bursts).
        deltas: list[float] = []
        if len(self.disc_hist.values) >= 2:
            prev = self.disc_hist.values[0]
            for v in self.disc_hist.values[1:]:
                d = v - prev
                deltas.append(d if d >= 0 else 0.0)
                prev = v
        opanel.disc_spark_values = deltas
        self.query_one('#footer', FooterBar).discoveries_n = (
            self.discoveries.total_count
        )


def main() -> int:
    import argparse
    import shutil
    parser = argparse.ArgumentParser(description='3D CA live dashboard (TUI).')
    parser.add_argument(
        '--stream', action='store_true',
        help='Stream-friendly mode: hide PIDs/host info for OBS streaming.',
    )
    args = parser.parse_args()
    if args.stream:
        os.environ['CA_DASHBOARD_STREAM'] = '1'
        # Re-bind module-level flag (it was read at import).
        global STREAM_MODE
        STREAM_MODE = True

    # Soft warning if the terminal is smaller than the design target.
    cols, rows = shutil.get_terminal_size((TARGET_COLS, TARGET_ROWS))
    if cols < TARGET_COLS or rows < TARGET_ROWS:
        print(
            f'[ca_dashboard] terminal is {cols}x{rows}; '
            f'designed for {TARGET_COLS}x{TARGET_ROWS}. '
            'Resize for the best layout (especially for streaming).',
        )
        time.sleep(1.2)

    app = CADashboard()
    app.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

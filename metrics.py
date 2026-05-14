"""
metrics.py — Performance metrics collection for parallel_processor.py

Tracks:
- Queue depths and capacity
- Worker busy/idle time
- Stage timings (parse/eval/write)
- Bottleneck diagnosis
"""

import time
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class WorkerStats:
    worker_id: int
    engine: str
    games_done: int = 0
    moves_done: int = 0
    busy_ms: float = 0.0
    idle_ms: float = 0.0
    last_active: float = field(default_factory=time.time)
    _last_state: str = 'idle'
    _state_start: float = field(default_factory=time.time)

    def mark_busy(self):
        now = time.time()
        if self._last_state == 'idle':
            self.idle_ms += (now - self._state_start) * 1000
        self._last_state = 'busy'
        self._state_start = now
        self.last_active = now

    def mark_idle(self):
        now = time.time()
        if self._last_state == 'busy':
            self.busy_ms += (now - self._state_start) * 1000
        self._last_state = 'idle'
        self._state_start = now

    @property
    def busy_pct(self) -> float:
        total = self.busy_ms + self.idle_ms
        return (self.busy_ms / total * 100) if total > 0 else 0.0


@dataclass
class QueueStats:
    name: str
    maxsize: int
    samples: deque = field(default_factory=lambda: deque(maxlen=100))

    def record(self, size: int):
        self.samples.append((time.time(), size))

    @property
    def current(self) -> int:
        return self.samples[-1][1] if self.samples else 0

    @property
    def avg(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s[1] for s in self.samples) / len(self.samples)

    @property
    def fill_pct(self) -> float:
        if self.maxsize == 0:
            return 0.0
        return self.current / self.maxsize * 100


class MetricsCollector:
    """
    Collects performance metrics for parallel processing pipeline.

    Usage:
        metrics = MetricsCollector()
        metrics.start()

        # In main thread:
        metrics.record_queue_size('game', queue.qsize())

        # In workers:
        metrics.worker_busy(worker_id)
        metrics.worker_idle(worker_id)

        # Periodically:
        print(metrics.summary())
    """

    def __init__(self, report_interval: float = 30.0):
        self.report_interval = report_interval
        self.start_time = time.time()

        self.queues: Dict[str, QueueStats] = {}
        self.workers: Dict[int, WorkerStats] = {}
        self.stage_timings: Dict[str, List[float]] = defaultdict(list)

        self.games_dispatched = 0
        self.games_completed = 0
        self.games_skipped = 0

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reporter_thread: Optional[threading.Thread] = None

    def register_queue(self, name: str, maxsize: int):
        with self._lock:
            self.queues[name] = QueueStats(name=name, maxsize=maxsize)

    def register_worker(self, worker_id: int, engine: str):
        with self._lock:
            self.workers[worker_id] = WorkerStats(worker_id=worker_id, engine=engine)

    def record_queue_size(self, name: str, size: int):
        with self._lock:
            if name in self.queues:
                self.queues[name].record(size)

    def worker_busy(self, worker_id: int):
        with self._lock:
            if worker_id in self.workers:
                self.workers[worker_id].mark_busy()

    def worker_idle(self, worker_id: int):
        with self._lock:
            if worker_id in self.workers:
                self.workers[worker_id].mark_idle()

    def record_stage_time(self, stage: str, duration_ms: float):
        with self._lock:
            self.stage_timings[stage].append(duration_ms)
            # Keep only last 1000 samples
            if len(self.stage_timings[stage]) > 1000:
                self.stage_timings[stage] = self.stage_timings[stage][-1000:]

    def increment_dispatched(self):
        with self._lock:
            self.games_dispatched += 1

    def increment_completed(self):
        with self._lock:
            self.games_completed += 1

    def increment_skipped(self):
        with self._lock:
            self.games_skipped += 1

    def start(self):
        """Start periodic reporting thread."""
        if self._reporter_thread is not None:
            return

        self._reporter_thread = threading.Thread(
            target=self._reporter_loop,
            daemon=True,
            name="metrics-reporter"
        )
        self._reporter_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._reporter_thread:
            self._reporter_thread.join(timeout=5)

    def _reporter_loop(self):
        while not self._stop_event.wait(self.report_interval):
            print(self.summary())

    def diagnose_bottleneck(self) -> str:
        """Identify the current bottleneck in the pipeline."""
        with self._lock:
            # Check queue depths
            game_q = self.queues.get('game')
            result_q = self.queues.get('result')

            if game_q and game_q.avg < game_q.maxsize * 0.1:
                return "parser_starved (game queue nearly empty - PGN reading is bottleneck)"

            if result_q and result_q.avg > result_q.maxsize * 0.8:
                return "result_queue_full (collector can't keep up)"

            # Check worker utilization
            lc0_workers = [w for w in self.workers.values() if w.engine == 'lc0']
            sf_workers = [w for w in self.workers.values() if w.engine == 'stockfish']

            if lc0_workers:
                avg_busy = sum(w.busy_pct for w in lc0_workers) / len(lc0_workers)
                if avg_busy < 50:
                    return f"lc0_underutilized (avg {avg_busy:.0f}% busy - GPU not saturated)"

            if sf_workers:
                avg_busy = sum(w.busy_pct for w in sf_workers) / len(sf_workers)
                if avg_busy < 50:
                    return f"stockfish_underutilized (avg {avg_busy:.0f}% busy - CPU not saturated)"

            # Check if game queue is full (backpressure)
            if game_q and game_q.fill_pct > 90:
                return "game_queue_full (workers can't keep up - need more workers or faster eval)"

            return "balanced (no clear bottleneck)"

    def summary(self) -> str:
        """Generate a human-readable summary of current metrics."""
        with self._lock:
            elapsed = time.time() - self.start_time
            rate = self.games_completed / elapsed if elapsed > 0 else 0.0

            lines = [
                f"\n{'='*70}",
                f"  METRICS SUMMARY (elapsed: {elapsed:.1f}s)",
                f"{'='*70}",
            ]

            # Overall progress
            lines.append(f"  Games: dispatched={self.games_dispatched}, "
                        f"completed={self.games_completed}, "
                        f"skipped={self.games_skipped}, "
                        f"rate={rate:.2f}/s")

            # Queue status
            if self.queues:
                lines.append(f"\n  Queues:")
                for q in self.queues.values():
                    lines.append(f"    {q.name:10s}: {q.current:4d}/{q.maxsize:4d} "
                                f"(avg={q.avg:.1f}, {q.fill_pct:.0f}% full)")

            # Worker status
            if self.workers:
                by_engine = defaultdict(list)
                for w in self.workers.values():
                    by_engine[w.engine].append(w)

                lines.append(f"\n  Workers:")
                for engine, workers in sorted(by_engine.items()):
                    avg_busy = sum(w.busy_pct for w in workers) / len(workers)
                    total_games = sum(w.games_done for w in workers)
                    lines.append(f"    {engine:10s}: {len(workers)} workers, "
                                f"avg {avg_busy:.0f}% busy, "
                                f"{total_games} games done")

                    # Show individual workers if any are idle
                    idle_workers = [w for w in workers if w.busy_pct < 50]
                    if idle_workers and len(workers) <= 10:
                        for w in idle_workers:
                            lines.append(f"      worker {w.worker_id}: {w.busy_pct:.0f}% busy")

            # Stage timings
            if self.stage_timings:
                lines.append(f"\n  Stage timings (avg ms):")
                for stage, timings in sorted(self.stage_timings.items()):
                    if timings:
                        avg = sum(timings) / len(timings)
                        lines.append(f"    {stage:15s}: {avg:.1f} ms (n={len(timings)})")

            # Bottleneck diagnosis
            bottleneck = self.diagnose_bottleneck()
            lines.append(f"\n  Bottleneck: {bottleneck}")
            lines.append(f"{'='*70}\n")

            return "\n".join(lines)

    def final_report(self) -> str:
        """Generate final report on shutdown."""
        with self._lock:
            elapsed = time.time() - self.start_time

            lines = [
                f"\n{'='*70}",
                f"  FINAL METRICS REPORT",
                f"{'='*70}",
                f"  Total runtime: {elapsed:.1f}s",
                f"  Games dispatched: {self.games_dispatched}",
                f"  Games completed: {self.games_completed}",
                f"  Games skipped: {self.games_skipped}",
                f"  Overall rate: {self.games_completed/elapsed:.2f} games/s",
            ]

            if self.workers:
                lines.append(f"\n  Worker breakdown:")
                for w in sorted(self.workers.values(), key=lambda x: (x.engine, x.worker_id)):
                    total_ms = w.busy_ms + w.idle_ms
                    lines.append(
                        f"    {w.engine:10s} worker {w.worker_id}: "
                        f"{w.games_done} games, "
                        f"{w.busy_pct:.0f}% busy ({total_ms/1000:.1f}s total)"
                    )

            lines.append(f"{'='*70}\n")
            return "\n".join(lines)

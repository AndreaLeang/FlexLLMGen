"""
power_monitor.py  (dual-socket update)
PowerSample now carries per-socket fields in addition to totals.
"""

import threading, time, csv
from dataclasses import dataclass
from typing import List, Optional
import pynvml
from rapl_reader import RaplReader


@dataclass
class PowerSample:
    timestamp:      float         # seconds since monitor start

    # ── aggregated (sum across all sockets) ──────────────────────────
    cpu_pkg_w:      float         # total pkg power  [W]
    cpu_dram_w:     float         # total DRAM power [W]

    # ── per-socket breakdown ──────────────────────────────────────────
    socket_pkg_w:   List[float]   # [socket0_pkg, socket1_pkg, ...]
    socket_dram_w:  List[float]   # [socket0_dram, socket1_dram, ...]

    # ── GPU ───────────────────────────────────────────────────────────
    gpu_w:          List[float]   # per-GPU via NVML [W]


class PowerMonitor:
    """
    Dual-socket aware power monitor.

    socket_ids=None   → auto-detect all RAPL sockets
    socket_ids=[0,1]  → explicit dual-socket
    socket_ids=[0]    → single socket (original behaviour)
    """

    def __init__(
        self,
        interval_ms:  int            = 100,
        gpu_indices:  List[int]      = None,
        socket_ids:   Optional[List[int]] = None,   # NEW
    ):
        self.interval_s  = interval_ms / 1000.0
        self.gpu_indices = gpu_indices or [0]
        self.socket_ids  = socket_ids           # None = auto-detect
        self.samples: List[PowerSample] = []
        self._stop      = threading.Event()
        self._thread    = None
        self._handles   = []
        self._rapl      = None

    def _init_hw(self):
        pynvml.nvmlInit()
        self._handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.gpu_indices
        ]
        self._rapl = RaplReader(socket_ids=self.socket_ids)

    def _loop(self, t0: float):
        while not self._stop.is_set():
            t     = time.perf_counter() - t0
            gpu_w = [pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                     for h in self._handles]

            # per-socket readings (preserves socket 0 / socket 1 split)
            per_socket   = self._rapl.read_power_per_socket()
            s_pkg_w      = [p for p, _ in per_socket]
            s_dram_w     = [d for _, d in per_socket]
            total_pkg    = sum(s_pkg_w)
            total_dram   = sum(s_dram_w)

            self.samples.append(PowerSample(
                timestamp     = t,
                cpu_pkg_w     = total_pkg,
                cpu_dram_w    = total_dram,
                socket_pkg_w  = s_pkg_w,
                socket_dram_w = s_dram_w,
                gpu_w         = gpu_w,
            ))
            time.sleep(self.interval_s)

    def start(self):
        self._init_hw(); self.samples.clear(); self._stop.clear()
        t0 = time.perf_counter()
        self._thread = threading.Thread(
            target=self._loop, args=(t0,), daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set(); self._thread.join()
        pynvml.nvmlShutdown(); return self

    def __enter__(self): return self.start()
    def __exit__(self, *_): self.stop()

    def summary(self) -> dict:
        s, n = self.samples, len(self.samples)
        if not n: return {}
        dur  = s[-1].timestamp - s[0].timestamp
        avg  = lambda vals: sum(vals) / n

        n_sockets = len(s[0].socket_pkg_w)
        n_gpus    = len(self.gpu_indices)

        pkg   = avg(x.cpu_pkg_w  for x in s)
        dram  = avg(x.cpu_dram_w for x in s)
        gpus  = [avg(x.gpu_w[i]  for x in s) for i in range(n_gpus)]

        # per-socket averages
        skt_pkg  = [avg(x.socket_pkg_w[i]  for x in s) for i in range(n_sockets)]
        skt_dram = [avg(x.socket_dram_w[i] for x in s) for i in range(n_sockets)]

        return dict(
            duration_s        = dur,
            n_samples         = n,
            n_sockets         = n_sockets,
            avg_cpu_pkg_w     = pkg,    energy_cpu_pkg_j     = pkg  * dur,
            avg_cpu_dram_w    = dram,   energy_cpu_dram_j    = dram * dur,
            avg_gpu_w         = gpus,   energy_gpu_j         = [g * dur for g in gpus],
            avg_socket_pkg_w  = skt_pkg,
            avg_socket_dram_w = skt_dram,
            energy_socket_pkg_j  = [p * dur for p in skt_pkg],
            energy_socket_dram_j = [d * dur for d in skt_dram],
        )

    def save_csv(self, path: str):
        n_sockets = len(self.samples[0].socket_pkg_w) if self.samples else 0
        n_gpus    = len(self.gpu_indices)

        skt_cols  = ([f"s{i}_pkg_w"  for i in range(n_sockets)] +
                     [f"s{i}_dram_w" for i in range(n_sockets)])
        gpu_cols  = [f"gpu{i}_w" for i in self.gpu_indices]

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_s",
                        "cpu_pkg_w", "cpu_dram_w"]   # totals
                       + skt_cols                       # per-socket
                       + gpu_cols)                      # GPU
            for x in self.samples:
                w.writerow([
                    f"{x.timestamp:.4f}",
                    f"{x.cpu_pkg_w:.2f}", f"{x.cpu_dram_w:.2f}",
                    *[f"{v:.2f}" for v in x.socket_pkg_w],
                    *[f"{v:.2f}" for v in x.socket_dram_w],
                    *[f"{g:.2f}" for g in x.gpu_w],
                ])
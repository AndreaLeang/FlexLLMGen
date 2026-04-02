"""
rapl_reader.py  (dual-socket update)
Reads RAPL energy counters for one or more CPU sockets and
aggregates them into combined pkg and DRAM power readings.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple


RAPL_BASE = Path("/sys/class/powercap/intel-rapl")


def detect_sockets() -> List[int]:
    """Return sorted list of available RAPL socket indices."""
    sockets = []
    for p in sorted(RAPL_BASE.iterdir()):
        # top-level entries look like intel-rapl:0, intel-rapl:1, ...
        parts = p.name.split(":")
        if len(parts) == 2 and parts[1].isdigit():
            sockets.append(int(parts[1]))
    return sockets


def _find_sub_domain(pkg_path: Path, name: str) -> Optional[Path]:
    for sub in pkg_path.iterdir():
        nf = sub / "name"
        if nf.exists() and nf.read_text().strip() == name:
            return sub
    return None


class _SocketReader:
    """Reads one physical CPU socket's RAPL pkg + DRAM counters."""

    def __init__(self, socket: int):
        self.socket = socket
        pkg = RAPL_BASE / f"intel-rapl:{socket}"
        if not pkg.exists():
            raise FileNotFoundError(
                f"Socket {socket} not found. Available: {detect_sockets()}")

        self._pkg_uj  = pkg / "energy_uj"
        self._pkg_max = int((pkg / "max_energy_range_uj").read_text())

        dram = _find_sub_domain(pkg, "dram")
        self._dram_uj  = (dram / "energy_uj") if dram else None
        self._dram_max = (
            int((dram / "max_energy_range_uj").read_text())
            if dram else None
        )

        # prime previous readings
        self._p_pkg, self._p_t = self._r_pkg()
        self._p_dram           = self._r_dram()

    def _r_pkg(self) -> Tuple[int, float]:
        return int(self._pkg_uj.read_text()), time.perf_counter()

    def _r_dram(self) -> int:
        return int(self._dram_uj.read_text()) if self._dram_uj else 0

    def read_power(self) -> Tuple[float, float]:
        """Returns (pkg_watts, dram_watts) since last call."""
        cur_pkg, cur_t = self._r_pkg()
        cur_dram       = self._r_dram()
        dt = cur_t - self._p_t
        if dt <= 0: return 0.0, 0.0

        dpkg  = (cur_pkg  - self._p_pkg)  % self._pkg_max
        ddram = (cur_dram - self._p_dram) % self._dram_max \
                if self._dram_max else 0

        pkg_w  = (dpkg  / 1e6) / dt
        dram_w = (ddram / 1e6) / dt if self._dram_uj else 0.0

        self._p_pkg, self._p_dram, self._p_t = cur_pkg, cur_dram, cur_t
        return pkg_w, dram_w


class RaplReader:
    """
    Multi-socket RAPL reader.

    Returns per-socket AND aggregated (sum) readings so callers
    can choose the granularity they need.

    socket_ids=None  →  auto-detect all available sockets
    socket_ids=[0]   →  single socket (original behaviour)
    socket_ids=[0,1] →  explicit dual-socket
    """

    def __init__(self, socket_ids: Optional[List[int]] = None):
        ids = socket_ids if socket_ids is not None else detect_sockets()
        if not ids:
            raise RuntimeError("No RAPL sockets found.")
        self.socket_ids = ids
        self._readers   = [_SocketReader(i) for i in ids]
        print(f"[RAPL] Using sockets: {ids}")

    def read_power(self) -> Tuple[float, float]:
        """
        Returns (total_pkg_watts, total_dram_watts) summed across all sockets.
        Use this as a drop-in replacement for the single-socket version.
        """
        total_pkg, total_dram = 0.0, 0.0
        for r in self._readers:
            pkg, dram = r.read_power()
            total_pkg  += pkg
            total_dram += dram
        return total_pkg, total_dram

    def read_power_per_socket(self) -> List[Tuple[float, float]]:
        """
        Returns [(pkg_w, dram_w), ...] one tuple per socket.
        Useful when you want to see socket 0 vs socket 1 separately,
        e.g. to isolate which socket is handling KV staging work.
        """
        return [r.read_power() for r in self._readers]

    @property
    def n_sockets(self) -> int:
        return len(self._readers)
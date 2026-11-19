"""Wait until >= 6 GB RAM are free (the author's loop rule), then run one controls GPU step; log to results/controls_wait_run.log.

    start_hidden_controls.ps1 controls_arm_b.py gen      # launched detached + hidden, survives the session closing
Checks every 60 s; before starting, needs 3 checks in a row at >= 6 GB. Gives up after 48 h (logged).
"""
import ctypes
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
LOG = HERE.parent / "results" / "controls_wait_run.log"
NEED_GB, STABLE, MAX_H = 6.0, 3, 48


class MEM(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_gb():
    m = MEM()
    m.dwLength = ctypes.sizeof(MEM)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30


def log(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")


def main():
    step = sys.argv[1:]
    log(f"waiting for >= {NEED_GB} GB free to run {' '.join(step)}")
    t0, ok, last = time.time(), 0, 0.0
    while ok < STABLE:
        if time.time() - t0 > MAX_H * 3600:
            log("gave up after 48 h - nothing run")
            return
        g = free_gb()
        ok = ok + 1 if g >= NEED_GB else 0
        if time.time() - last > 1800:
            log(f"free {g:.1f} GB")
            last = time.time()
        time.sleep(60)
    log(f"start (free {free_gb():.1f} GB)")
    with open(LOG, "a", encoding="utf-8") as fh:
        r = subprocess.run([sys.executable, "-W", "ignore", *step], cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
    log(f"exit {r.returncode}")


if __name__ == "__main__":
    main()

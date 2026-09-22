"""Bootstrap executed inside the Colab kernel via `colab exec`.

Spawns the worker as a detached process so it survives this exec call.
Idempotent: a live worker is not started twice.
"""
import os
import subprocess
import time

STATUS = "/content/rvc_work/status.json"
PIDFILE = "/content/rvc_work/worker.pid"


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


pid = None
if os.path.exists(PIDFILE):
    try:
        pid = int(open(PIDFILE).read().strip())
    except ValueError:
        pid = None

if pid and _alive(pid):
    print(f"BOOTSTRAP::already_running pid={pid}")
else:
    os.makedirs("/content/rvc_work", exist_ok=True)
    # clear stale status so the orchestrator doesn't read a dead run's result
    if os.path.exists(STATUS):
        os.remove(STATUS)
    out = open("/content/rvc_work/worker.out", "a", buffering=1)
    p = subprocess.Popen(
        ["python", "/content/worker.py"],
        stdout=out, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    open(PIDFILE, "w").write(str(p.pid))
    print(f"BOOTSTRAP::started pid={p.pid}")

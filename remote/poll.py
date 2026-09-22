"""Polled via `colab exec`: prints one STATUS::<json> line for the orchestrator."""
import json
import os

STATUS = "/content/rvc_work/status.json"
LOG = "/content/rvc_work/worker.log"

if os.path.exists(STATUS):
    print("STATUS::" + open(STATUS).read().replace("\n", " "))
elif os.path.exists(LOG):
    lines = open(LOG).read().strip().splitlines()[-3:]
    print("STATUS::" + json.dumps({"phase": "boot", "tail": lines}))
else:
    print("STATUS::{\"phase\": \"no_status\"}")

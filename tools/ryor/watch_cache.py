"""Audit new recordings until all gallery setups pass, then exit. No cloud compute is launched here."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
while True:
    subprocess.run([sys.executable, str(ROOT / 'tools/ryor/audit_cache.py')], check=True)
    results = json.loads((ROOT / 'handoff/ryor_evidence/cache_audit.json').read_text())
    if results and all(r['valid'] for r in results.values()):
        print('All default recordings passed. Cache watcher is stopping.', flush=True)
        break
    time.sleep(60)  # Poll a slow batch at minute cadence without launching duplicate jobs.

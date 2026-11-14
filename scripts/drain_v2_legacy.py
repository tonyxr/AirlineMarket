"""R2: freeze legacy dispatch, retain the active child, close after its exit.

This control script does not change captured code. The suspended coordinator
stays the child's parent. Its existing wall/RSS limits are watched here. A
native worker's completed receipt is retained; limit-only receipts use the
legacy schema and include the control-script provenance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone

import psutil


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.drain-tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def identity(record):
    try:
        p = psutil.Process(record['pid'])
        if abs(p.create_time() - record['created']) > .1:
            raise ValueError('PID identity changed')
        return p
    except psutil.NoSuchProcess:
        return None


def alive(p):
    try:
        return p is not None and p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def drain(job, output):
    root, attempt = Path(job['root']), Path(job['attempt'])
    parent, child = identity(job['coordinator']), identity(job['worker'])
    if not alive(parent):
        raise ValueError('coordinator absent before dispatch stop')
    if not alive(child) or child.ppid() != parent.pid:
        raise ValueError('active worker identity changed; refresh before draining')
    worker_children = [p.pid for p in parent.children() if '--parent' in p.cmdline()]
    if worker_children != [child.pid]:
        raise ValueError('expected exactly the declared active worker')
    os.kill(parent.pid, signal.SIGSTOP)
    record = {**job, 'status': 'draining_v2', 'dispatch_suspended': True,
              'guard_pid': os.getpid(), 'guard_created': psutil.Process().create_time(),
              'started_at': datetime.now(timezone.utc).isoformat()}
    atomic(output, record)
    cell = json.loads((attempt / 'config.json').read_text())
    started = datetime.fromisoformat(job['worker_started_at']).timestamp()
    peak, limit = 0, None
    while alive(child):
        rss = child.memory_info().rss
        progress = attempt / 'progress.json'
        if progress.exists():
            try:
                rss += sum(x['rss_bytes'] for x in json.loads(progress.read_text()).get('owned_children', []))
            except (ValueError, KeyError):
                pass
        peak = max(peak, rss)
        elapsed = time.time() - started
        if elapsed >= float(cell['limits'].get('seconds', 14400)):
            limit = 'time_limit'
        elif peak > float(cell['limits'].get('rss_gb', 12)) * 1024**3:
            limit = 'memory_limit'
        if limit:
            os.killpg(child.pid, signal.SIGTERM)
            for _ in range(50):
                if not alive(child):
                    break
                time.sleep(.1)
            if alive(child):
                os.killpg(child.pid, signal.SIGKILL)
            while alive(child):
                time.sleep(.1)
            break
        time.sleep(.25)
    # No worker remains to lose its parent; a frozen coordinator cannot dispatch.
    if identity(job['coordinator']) is not None:
        os.kill(parent.pid, signal.SIGKILL)
    receipt = attempt / 'COMPLETE.json'
    if receipt.exists():
        result = json.loads((attempt / 'results.json').read_text())
        seal = json.loads(receipt.read_text())
        if hashlib.sha256((attempt / 'results.json').read_bytes()).hexdigest() != seal['results_sha256']:
            raise ValueError('completed native result hash differs')
    else:
        result = {'status': limit or 'failed', 'config_hash': cell['config_hash'],
                  'parameters': cell['parameters'], 'phase': cell['phase'], 'value': None,
                  'wall_seconds': time.time() - started, 'peak_rss_bytes': peak,
                  'reason': limit or 'worker exited without a completed receipt',
                  'control_provenance': 'scripts/drain_v2_legacy.py; R2 dispatch drain'}
        atomic(attempt / 'results.json', result)
        atomic(receipt, {'config_hash': cell['config_hash'], 'status': result['status'],
                        'results_sha256': hashlib.sha256((attempt / 'results.json').read_bytes()).hexdigest()})
    atomic(attempt / 'status.json', {'status': result['status'], 'cell_id': cell['cell_id'],
                                  'completed_at': datetime.now(timezone.utc).isoformat(),
                                  'control_provenance': 'R2 dispatch drain'})
    record.update(status='stopped_v2', worker_result=result['status'],
                  finished_at=datetime.now(timezone.utc).isoformat(),
                  native_receipt_retained='control_provenance' not in result,
                  observed_peak_rss_bytes=peak)
    atomic(output, record)
    status = json.loads((root / 'status.json').read_text())
    status.update(status='stopped_v2', stopped_at=record['finished_at'],
                  stop_reason='v2.0 Section 3 / owner adoption; current cell finished',
                  r2_control_record=str(output))
    atomic(root / 'status.json', status)


def main():
    p = argparse.ArgumentParser(); p.add_argument('job'); p.add_argument('--detach', action='store_true')
    args = p.parse_args(); path = Path(args.job).resolve()
    if args.detach:
        with path.with_suffix('.log').open('a') as log:
            proc = subprocess.Popen([os.sys.executable, '-u', str(Path(__file__).resolve()), str(path)],
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        print(json.dumps({'guard_pid': proc.pid, 'job': str(path)})); return
    drain(json.loads(path.read_text()), path.with_suffix('.state.json'))


if __name__ == '__main__':
    main()

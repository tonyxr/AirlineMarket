"""R2 controls: the current child completes; queued work cannot launch."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import psutil
import pytest

spec = importlib.util.spec_from_file_location('drain_control', Path('scripts/drain_v2_legacy.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fixture_job(tmp_path, seconds):
    root = tmp_path / 'campaign'; attempt = root / 'cells' / 'x' / 'attempt-0001'
    attempt.mkdir(parents=True)
    (root / 'status.json').write_text('{"status":"running"}')
    cell = {'cell_id':'x', 'config_hash':'abc', 'phase':'R2', 'parameters':{},
            'limits':{'seconds':seconds, 'rss_gb':1}}
    (attempt / 'config.json').write_text(json.dumps(cell))
    child = tmp_path / 'child.py'
    child.write_text('import time,json,hashlib,sys\nfrom pathlib import Path\np=Path(sys.argv[1])\ntime.sleep(.7)\nr=p/"results.json";r.write_text(json.dumps({"status":"completed","value":7}))\n(p/"COMPLETE.json").write_text(json.dumps({"results_sha256":hashlib.sha256(r.read_bytes()).hexdigest()}))\n')
    parent = tmp_path / 'parent.py'
    parent.write_text('import subprocess,sys,os,json,time\nfrom pathlib import Path\np=subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2],"--parent",str(os.getpid())],start_new_session=True)\nPath(sys.argv[3]).write_text(str(p.pid))\np.wait()\nPath(sys.argv[4]).write_text("queued work started")\ntime.sleep(30)\n')
    ready = tmp_path / 'ready'; queued = tmp_path / 'queued'
    proc = subprocess.Popen([os.sys.executable,str(parent),str(child),str(attempt),str(ready),str(queued)],start_new_session=True)
    for _ in range(100):
        if ready.exists(): break
        time.sleep(.01)
    worker = psutil.Process(int(ready.read_text()))
    job = {'root':str(root),'attempt':str(attempt),
           'coordinator':{'pid':proc.pid,'created':psutil.Process(proc.pid).create_time()},
           'worker':{'pid':worker.pid,'created':worker.create_time()},
           'worker_started_at':m.datetime.now(m.timezone.utc).isoformat()}
    return job, proc, queued


def test_current_worker_finishes_and_queue_stays_closed(tmp_path):
    job, proc, queued = fixture_job(tmp_path, 10)
    out = tmp_path / 'state.json'
    m.drain(job, out); proc.wait(timeout=5)
    result = json.loads(out.read_text())
    assert result['status'] == 'stopped_v2'
    assert result['native_receipt_retained']
    assert not queued.exists()
    assert json.loads((Path(job['attempt']) / 'results.json').read_text())['value'] == 7


def test_original_wall_limit_is_preserved(tmp_path):
    job, proc, queued = fixture_job(tmp_path, .1)
    m.drain(job, tmp_path / 'state.json'); proc.wait(timeout=5)
    result = json.loads((Path(job['attempt']) / 'results.json').read_text())
    assert result['status'] == 'time_limit' and result['value'] is None
    assert not queued.exists()


def test_identity_mismatch_raises_without_signaling():
    with pytest.raises(ValueError, match='identity changed'):
        m.identity({'pid':os.getpid(), 'created':0})

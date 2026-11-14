"""A.15/V10: admit scoped event-data streaming after R2 reservations release."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'config/experiments/v2/A15_E3_scope.json'
STATE=ROOT/'results/A15_tooling/deferred_launch.json'
DRAINS=[ROOT/'results/v2_transition'/('drain_'+name+'.state.json') for name in ('F1','F1_history_next_frontier','F5')]


def eligible(records):
    """A finished R2 record releases its old reservation; no partial-tree inference."""
    return all(r.get('status')=='stopped_v2' for r in records)


def main():
    with STATE.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        digest=hashlib.sha256(CONFIG.read_bytes()).hexdigest()
        state={'plan_id':'A.15/V10','status':'waiting_for_R2_resource_release','pid':os.getpid(),
               'created':psutil.Process().create_time(),'config':str(CONFIG),'config_sha256':digest,
               'market_months':7,'pre_event_marketing_months':4,'workers':1,'reservation_budget_gb':.5,
               'reason':'Wait for R2 drains. S1 reserves10.5GiB,F4reserves1GiB,eventstream0.5GiB,total12GiB. No source,month or row reduction.'}
        STATE.write_text(json.dumps(state,indent=2)+'\n')
        while not eligible([json.loads(p.read_text()) for p in DRAINS]):time.sleep(10)
        for path in DRAINS:
            record=json.loads(path.read_text());pid=record['coordinator']['pid']
            if psutil.pid_exists(pid):
                p=psutil.Process(pid)
                if abs(p.create_time()-record['coordinator']['created'])<.1 and p.status()!=psutil.STATUS_ZOMBIE:
                    raise RuntimeError('R2 coordinator still live')
        if hashlib.sha256(CONFIG.read_bytes()).hexdigest()!=digest:raise ValueError('queued A15config changed')
        p=subprocess.run([sys.executable,'-m','airline_market.experiments.settled_study_campaign',
                          '--config',str(CONFIG),'--detach'],cwd=ROOT,check=True,capture_output=True,text=True)
        state.update(status='launched',launch_record=json.loads(p.stdout),launched_at=time.time())
        STATE.write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':main()

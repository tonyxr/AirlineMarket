"""A.15/V10: finish the two explicitly authorized pending acquisition jobs."""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[2]
state=root/'results/data_acquisition/v2_followthrough.status.json'
upstream=root/'results/data_acquisition/v2_section9_2_db1c_market.status.json'
state.write_text(json.dumps({'plan_id':'A.15/V10','status':'waiting_for_serial_db1c_queue'})+'\n')
while not upstream.exists() or json.loads(upstream.read_text()).get('status')!='finished':
    time.sleep(10)
for manifest in ['v2_section9_2_july_fresh.json','v2_section9_2_db1b.json']:
    state.write_text(json.dumps({'plan_id':'A.15/V10','status':'running','manifest':manifest})+'\n')
    subprocess.run([sys.executable,'-u',str(root/'scripts/data_acquisition/download.py'),
                    str(root/'results/data_acquisition'/manifest),'--workers','1'],cwd=root,check=True)
state.write_text(json.dumps({'plan_id':'A.15/V10','status':'finished'})+'\n')

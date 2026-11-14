"""Audit the sealed scaling grid and fit conditional wall/CPU log-log slopes."""
from pathlib import Path
import argparse
import json
import numpy as np
from airline_market.campaign.artifacts import atomic_json,file_hash,utc_now
from airline_market.core.rng import StreamRegistry


def summarize(root, bootstrap=2000):
    """Bootstrap common cohort indices; retain machine-contention limitations."""
    results=json.loads((root/'results.json').read_text())['cells']
    groups={}
    provenance=[]
    for path in root.glob('cells/*/attempt-*/COMPLETE.json'):
        receipt=json.loads(path.read_text())
        result_path=path.with_name('results.json')
        if file_hash(result_path)!=receipt['results_sha256']:
            raise ValueError('damaged scaling receipt')
        result=json.loads(result_path.read_text())
        if result['status']!='completed':
            continue
        timing=path.with_name('timing_rows.json')
        rows=json.loads(timing.read_text())
        if len(rows)!=64 or [r['seed'] for r in rows]!=list(range(200000,200064)):
            raise ValueError('scaling cohort alignment differs from registered design')
        p=result['parameters']
        groups.setdefault((p['n_carriers'],p['periods']),{})[p['n_spokes']]=rows
        provenance.append({'result':str(result_path),'result_sha256':file_hash(result_path),
                           'timing_sha256':file_hash(timing)})
    rng=StreamRegistry(200020,'validation/scaling_slope_bootstrap').generator()
    indices=rng.integers(0,64,size=(bootstrap,64))
    slopes=[]
    for (carriers,periods),sizes in sorted(groups.items()):
        levels=sorted(sizes)
        if levels!=[2,4,8,16,32,50]:
            raise ValueError('incomplete scaling spoke catalog')
        x=np.log(levels)
        weights=(x-x.mean())/np.sum((x-x.mean())**2)
        for field in ['seconds','cpu_seconds']:
            costs=np.array([[r[field] for r in sizes[k]] for k in levels])
            point=float(weights @ np.log(costs.mean(axis=1)))
            replicates=np.log(costs[:,indices].mean(axis=2)).T @ weights
            slopes.append({'carriers':carriers,'periods':periods,'cost_field':field,
                           'slope':point,'bootstrap95':np.quantile(replicates,[0.025,0.975]).tolist(),
                           'spokes':levels,'mean_costs':costs.mean(axis=1).tolist()})
    result={'updated_at':utc_now(),'completed_cells':len(provenance),
            'timing_cohorts':sum(r.get('n_cohorts',0) for r in results if r['status']=='completed'),
            'registered_criterion':json.loads((root/'largest_criterion.json').read_text()),
            'slopes':slopes,'bootstrap_replicates':bootstrap,'provenance':provenance,
            'scope':'OLS log(mean cost) versus log(spokes), fixed carrier count/horizon. Common64 validation cohort indices resampled jointly. Wall and CPU reported separately. CIs describe cohort variability conditional on the observed workload; contention/time drift is not randomized or included. These are not uncontended hardware limits, and passing all72cells does not pass the2second criterion.'}
    atomic_json(root/'scaling_analysis.json',result)
    return {k:v for k,v in result.items() if k not in ['provenance','slopes','registered_criterion']}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    print(json.dumps(summarize(parser.parse_args().root),indent=2))

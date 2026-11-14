"""R4: regenerate the owner status from saved actual-process and campaign metadata."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def counts(root):
    result=Counter()
    for cell in (Path(root)/'cells').glob('*'):
        receipts=sorted(cell.glob('attempt-*/COMPLETE.json'))
        if receipts:result[json.loads(receipts[-1].read_text())['status']]+=1
    return dict(result)


def main():
    snap=json.loads(Path(sys.argv[1]).read_text())
    path=Path('results/monitor/execution_state.json');state=json.loads(path.read_text())
    registry=json.loads(Path('results/registry_v2.json').read_text())
    registered_campaigns=len(registry['campaigns'])
    registered_receipts=sum(len(c['cells']) for c in registry['campaigns'])
    now=datetime.now(timezone.utc).isoformat();v=state['v2'];tasks=v['tasks']
    drains=[json.loads(p.read_text()) for p in Path('results/v2_transition').glob('drain_*.state.json')]
    pending=sum(r['status']!='stopped_v2' for r in drains)
    f5_draining=any(r['campaign']=='F5' and r['status']!='stopped_v2' for r in drains)
    s1path=Path(state['v2'].get('S1_admission_state','results/v2_transition/S1_deferred_launch.json'))
    s1=json.loads(s1path.read_text()) if s1path.exists() else {'status':'registered'}
    i4=counts(state['phase1']['F4_I4']['out']);s1counts=counts('results/S1/20261113_round1_v2')
    event=state['v2'].get('event_data_processing',{})
    event_path=Path(event.get('admission_state','results/A15_tooling/deferred_launch.json'))
    event_admission=json.loads(event_path.read_text()) if event_path.exists() else {'status':'not_registered'}
    design=v.get('S1_value_design_analysis',{})
    design_number=f"; design rank {design['primary_rank']}/{design['primary_columns']}" if design else ''
    s1_label='Queued' if s1['status']=='waiting_for_F1_reservation_release' else s1['status'].replace('_',' ')
    event_label='Queued' if event_admission['status']=='waiting_for_F1_reservation_release' else event_admission['status'].replace('_',' ')
    memory_note='S1 then shares memory with F5.' if f5_draining else 'F5 is stopped; S1 then has a 10.5 GiB admission budget.'
    if pending==0:
        memory_note='All R2 drains are closed. S1 admission is 10.5 GiB; F4 plus saved-policy analysis reserve 1.5 GiB.'
    s1_next='Start after F1 releases memory' if pending else 'Finish exact arms; join new saved results'
    source_status=Path('results/S1/20261113_round1_v2/status.json')
    source_finished=source_status.exists() and json.loads(source_status.read_text()).get('status')=='finished'
    if source_finished:
        s1_label='Source complete'
        s1_next='Report empty eligible set; S2 default open'
        memory_note='S1 source reservation released. F4 reserves 1 GiB; analysis uses 0.5 GiB only when admitted.'
    event_next='Start after F1 releases memory' if pending else 'Finish corrected full extraction'
    if event.get('status')=='completed scoped extraction':
        event_label='Completed';event_next='Compile dated event network and reconcile sources'
    event_number='All 7 Market months; 4 pre-event marketing months'
    network=v.get('event_network_compiler',{})
    if network:
        event_label=network['status']
        event_number=f"8 schedule months; {network['physical_resources']:,} resources"
        event_next='Reconcile seats and fare/passenger totals; build model'
    seats=v.get('event_seat_comparison',{})
    if seats:
        event_label=seats['status']
        event_number=f"{seats['join_counts']['matched']:,} matched; {seats['performed_mismatches']:,} departure differences"
        event_next='Trace differences; fare/passenger and native model checks'
    trace=v.get('event_departure_trace',{})
    if trace:
        event_label=trace['status']
        event_number=f"{trace['outer_rows']:,} traced; {trace['performed_mismatches']:,} differences unresolved"
        event_next='Fare/passenger definitions and dated native model checks'
    moments=v.get('event_fare_moments',{})
    if moments:
        event_label=moments['status']
        event_number=f"{moments['market_records']:,} records; {moments['groups']:,} raw groups"
        event_next='Resolve raw flag meaning; native moments and dated models'
    metrics=v.get('S1_metrics',{})
    metrics_counts=counts(metrics['batch_root']) if metrics.get('batch_root') else {}
    metrics_finished=bool(metrics.get('batch_root') and (Path(metrics['batch_root'])/'status.json').exists() and json.loads((Path(metrics['batch_root'])/'status.json').read_text()).get('status')=='finished')
    metrics_next='Admit new ready settings; exclude prior IDs' if metrics_finished else 'Finish admitted batch; review new receipts'
    if metrics_finished and metrics.get('total_registered_metrics_settings')==52:
        metrics_next='Retain limits; no matching reruns'
    prior_metric_counts=[counts(root) for root in metrics.get('prior_batch_roots',[])]
    metric_complete=metrics_counts.get('completed',0)+sum(c.get('completed',0) for c in prior_metric_counts)
    metric_limits=metrics_counts.get('memory_limit',0)+sum(c.get('memory_limit',0) for c in prior_metric_counts)
    metrics_number=(f"{metric_complete} complete; {metric_limits} limits; {metrics.get('registered_batch_settings',0)} jobs in last batch" if metrics_counts or prior_metric_counts else 'Pairwise MI and first fare difference')
    data=[]
    for p in Path('results/data_acquisition/files').glob('V2*.json'):
        r=json.loads(p.read_text());data.append(r['status'])
    dc=dict(Counter(data));acq_label='Acquired' if state['v2'].get('scoped_acquisition_complete') else 'Acquiring';available=snap['memory']['available_bytes']/1024**3
    scientific=[p for p in snap['processes'] if '--parent' in p['command']]
    observed_pids={p['pid'] for p in snap['processes']}
    for key,e in state['phase1'].items():
        if e.get('out'):
            if e.get('coordinator_pid') is not None:
                e['coordinator_observed']=e['coordinator_pid'] in observed_pids
            e['observed_counts']=counts(e['out'])
            active=[]
            for process in scientific:
                cmd=process['command'];attempt=Path(cmd[cmd.index('--parent')-1])
                if not str(attempt).startswith(str(Path(e['out']).resolve())):continue
                progress=attempt/'progress.json'
                data=json.loads(progress.read_text()) if progress.exists() else None
                active.append({'pid':process['pid'],'directory':str(attempt),'progress':data,'observed_process':process})
            e['active_cells']=active;e['observed_at']=snap['updated_at']
            if active:e['observed_counts']['running']=len(active)
            if key=='F4_I4':
                e['cells_total']=len(json.loads((Path(e['out'])/'cells.json').read_text()))
                e['unstarted']=max(0,e['cells_total']-sum(e['observed_counts'].values()))
                if v.get('latest_V6_review'):
                    e['latest_completed_cell_review']=v['latest_V6_review']
                    saved_review=json.loads(Path(v['latest_V6_review']).read_text())
                    if saved_review.get('frames')==384 and 'provisional_criteria_pass' in saved_review:
                        e['latest_new_accuracy_review']=v['latest_V6_review']
            e['latest_progress']=active[0]['progress'] if len(active)==1 else None
    v.update(heartbeat_observed_at=now,screen_status=s1['status'],screen_counts=s1counts,
             R2_pending_current_cells=pending,data_acquisition_counts=dc)
    state['updated_at']=now;path.write_text(json.dumps(state,indent=2)+'\n')
    dated=v.get('dated_reserve_release',{})
    policy_status=dated.get('status',v.get('trigger_execution',v.get('reserved_seats',{})).get('status','Pending'))
    policy_number='Per-leg exact 24h UTC; native rescue/reserve/trigger' if dated else 'Rescue, reserve, public-trigger execution'
    policy_next='Dated eligibility; mechanical benchmark and planners' if dated else 'Build mechanical benchmark; dates and planners'
    dated_rescue=v.get('dated_rescue_reentry',{})
    rebooking_status=dated_rescue.get('status',v.get('rebooking',v.get('disruption_exit',{})).get('status','Pending'))
    rebooking_number='UTC eligibility; original lineages; censored timing summaries' if dated_rescue else 'Original-passenger conservation and native refunds'
    rebooking_next='Compose dated policies; broader shocks and planning' if dated_rescue else 'Add broader shocks and event-aware planning'
    if dated_rescue:
        policy_next='Compose dated policies; mechanical benchmark and planners'
    composition=v.get('dated_rescue_reserve_composition',{})
    if composition:
        policy_status=composition['status']
        policy_number='UTC eligibility + held seats; exact per-leg 24h release'
        policy_next='Trigger composition; mechanical benchmark and planners'
        rebooking_status='Dated rescue/reserve verified'
        rebooking_next='Broader shocks, welfare and event-aware planning'
    triggered=v.get('triggered_dated_execution',{})
    if triggered:
        policy_status=triggered['status']
        policy_number='Public trigger posting + UTC rescue/held-seat rules'
        policy_next='B2/public reports; rationing, welfare and planners'
    rationing=v.get('static_rationing',{})
    native_pool=v.get('native_ordinary_pool',{})
    cap_evidence=v.get('native_cap_evidence',{})
    native_admission=v.get('native_priority_admission',{})
    potential_lineage=v.get('potential_request_lineage',{})
    if potential_lineage:
        rebooking_status=potential_lineage['status']
        rebooking_number='Potential slots + native ranks; dated native proof'
        rebooking_next='Shared shock envelopes; welfare and planners'
    shared_demand=v.get('shared_poisson_demand',{})
    if shared_demand:
        rebooking_status=shared_demand['status']
        rebooking_number='Shared demand envelope + potential lineage'
        rebooking_next='Physical shocks; dated adapters; welfare/planners'
    temporary=v.get('temporary_capacity_windows',{})
    if temporary:
        rebooking_status=temporary['status']
        rebooking_number='Installed-seat loss/recovery; native refunds/lineage'
        rebooking_next='Dated shocks/grid; welfare and planners'
    dated_shocks=v.get('dated_B1_shock_adapters',{})
    if dated_shocks:
        rebooking_status=dated_shocks['status']
        rebooking_number='Exact UTC loss/exit; 1–3 elapsed-day windows'
        rebooking_next='Demand/composite dates; B1 grid/split/event anchoring'
    dated_demand=v.get('dated_B1_demand_composite',{})
    if dated_demand:
        rebooking_status=dated_demand['status']
        rebooking_number='UTC physical + surge/finite mix + hurricane re-entry'
        rebooking_next='B1 grid/split, mix-duration choices and event anchoring'
    dated_family=v.get('dated_B1_shared_family',{})
    if dated_family:
        rebooking_status=dated_family['status']
        rebooking_number='One shared bank: five dated types + control'
        rebooking_next='B1 grid/split/anchors; compatible planners'
    decomposition=v.get('B2_decomposition_arithmetic',{})
    decomposition_row=(f"| B2 / H2.1 | Spike decomposition | {decomposition['status']} | 8 subsets + no-shock; residual retained | Channel/metric adapters; paired inference |\n" if decomposition else '')
    panel=v.get('B2_native_fare_panel',{})
    if panel:
        decomposition_row=(f"| B2 / H2.1 | Spike measurement | {panel['status']} | SHA-linked postings; explicit weights / 72h | Economic channels; primary metrics and inference |\n")
    event_counts=v.get('B1_event_reported_counts',{})
    event_counts_row=(f"| B1 / A.15 | Event magnitudes | Reported-flight adapter verified | E3: {event_counts['rate_null_cells']:,} rate cells null | Identify seat-loss denominator; no imputation |\n" if event_counts else '')
    allocation_compare=v.get('native_static_allocation_comparison',{})
    rationing_key=('Native priority + static score comparison' if allocation_compare else 'Re-entry first; full-period ordinary priority')
    rationing_next=('Efficient admission; lifecycle welfare and planners' if allocation_compare else 'Other groups; efficient allocation; planners')
    welfare=v.get('ordinary_welfare_accounting',{})
    welfare_number=('Declared one-person policy payoff; CS open' if v.get('finite_reentry_policy_payoff') else 'Own-first attempt gain; lifecycle open' if v.get('ordered_reentry_logit') else 'Declared UTC unreserved spans; lifecycle open' if v.get('recurrent_unreserved_spells') else 'Payoff timing definitions; native CS/points' if v.get('single_commitment_payoff_logit') else 'Original-person reservation points; CS/forecasts' if v.get('ordinary_reservation_census') else 'Finite forecast stock law; cohort/accounts' if v.get('ordinary_poisson_forecast') else 'Conditional Poisson period; cohort/accounts' if v.get('ordinary_poisson_expectation') else 'Conditional CS; native ordinary-request strata')
    text=f'''# AirlineMarket status — {now[:16].replace('T',' ')} UTC
Plan v2.0 | Phase {v.get('phase','R')} / V build | Next decision: D1, November 23

| ID | Item | Status | Key number | Next action |
|---|---|---|---|---|
| R1 | Adopt v2 | Done | PDF hash registered; monitor updated | Follow two-paper plan |
| R2 | Stop obsolete queues | {'Draining' if pending else 'Stopped'} | {pending} current cell{'s' if pending!=1 else ''} {'remain' if pending!=1 else 'remains'}; dispatch closed | {'Let current work finish under original limits' if pending else 'Keep stopped queues closed'} |
| R3 | Carry-over registry | Indexed | {registered_campaigns} campaigns; {registered_receipts} receipts | Add new terminal receipts |
| R4 | Owner status | Live | This page | Regenerate each heartbeat |
| R5 | Screen parameters/diagnostics | Verified | 8 MR7/MR9 models; 10k cohorts; 17 checks pass | Retain default-byte compatibility |
| S1 | Round 1 | {s1_label} | 52 settings × 5 arms; {s1counts.get('solved',0)} solved{design_number} | {s1_next} |
| S1 | Saved-policy metrics | {metrics.get('status','Pending')} | {metrics_number} | {metrics_next} |
| V6 | I4 particles | Continuing | {i4.get('completed',0)} accuracy complete; {i4.get('failed',0)} strict failures; {i4.get('time_limit',0)} timeout | Finish registered I4 queue |
| A.8 / B0 / V9 | Caps and rationing | {native_admission.get('status',cap_evidence.get('status',native_pool.get('status',rationing.get('status',v.get('cap_execution',{}).get('status','Pending')))))} | {rationing_key} | {rationing_next} |
| A.5 / B0 / V9 | Exit and rebooking | {rebooking_status} | {rebooking_number} | {rebooking_next} |
| A.14 / B0 / V9 | Policy components | {policy_status} | {policy_number} | {policy_next} |
{decomposition_row}{event_counts_row}| A.10 / B0 / V9 | Welfare accounting | {welfare.get('status','Pending')} | {welfare_number} | Lifecycle CS and denial/allocation comparisons |
| A.15 / V10 | Scoped Paper 2 inputs | {acq_label} | {dc.get('downloaded',0)} new files saved | Reuse prepared banks; native/date adapters |
| A.15 / V10 | Event pipeline | {event_label} | {event_number} | {event_next} |

Material limit: all 52 S1 settings are structurally implausible under both-carrier 70% load; owner explicitly retained them. Their exact value comparisons proceed unchanged.
Owner decisions needed: none now. D1 is due November 23.
Machine: {len(scientific)} scientific worker{'s' if len(scientific)!=1 else ''}; {available:.2f} GiB available at {snap['updated_at'][:16]} UTC. {memory_note} No exact-tree ETA.
'''
    Path('STATUS.md').write_text(text)
    print(json.dumps({'updated_at':now,'R2_pending_cells':pending,'S1':s1['status'],'F4_I4':i4,'new_data_counts':dc}))


if __name__=='__main__':main()

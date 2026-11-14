"""Compare represented ticket-path boardings with allocated segment passengers."""
from pathlib import Path
import argparse
import pandas as pd
from airline_market.campaign.artifacts import atomic_json,file_hash


def audit(targets,out):
    """Keep the source-coverage residual explicit; never impute it as identified demand."""
    tickets=pd.read_csv(targets/'q2_passenger_targets.csv')
    capacity=pd.read_csv(targets/'q2_capacity_targets.csv')
    tickets=tickets[tickets.model_path_eligible & tickets.path_network.isin(['UA','AA'])]
    if not capacity.month.isin([4,5,6]).all() or not (capacity.year==2022).all():
        raise ValueError('capacity targets must be2022Q2')
    rows=[]
    for carrier in ['UA','AA']:
        group=tickets[tickets.path_network==carrier]
        boardings=float((group.passengers*(1+group.hub_connecting.astype(int))).sum())
        segment=float(capacity.loc[capacity.marketing_carrier==carrier,'allocated_passengers'].sum())
        rows.append({'carrier':carrier,'represented_Q2_ticket_journeys':float(group.passengers.sum()),
                     'represented_ticket_leg_boardings':boardings,
                     'allocated_T100_Q2_passenger_boardings':segment,
                     'represented_minus_T100':boardings-segment,
                     'represented_to_T100':boardings/segment})
    result={'rows':rows,
            'scope':'Coverage diagnostic only. Eligible paths are one hub nonstop or exactly two coupons viaORD, as defined by scoped_tables. Compare their implied boardings with allocated T100 segment passengers. Source/sample/calendar/marketing allocation differences remain. The residual is not automatically attributed to outside-pool demand, imputed or rescaled. Audit off-scope/interline paths before using arrival/conversion parameters to fit all leg loads.',
            'input_hashes':{str(targets/n):file_hash(targets/n) for n in ['q2_passenger_targets.csv','q2_capacity_targets.csv']}}
    atomic_json(out,result)
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('targets',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    print(audit(args.targets,args.out))

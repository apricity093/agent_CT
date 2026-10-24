"""Generate independent P3 public observation cases without changing historical data."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from inv_framework.benchmarks.ct_cases import load_ct_case,write_ct_case
from inv_framework.measurement import simulate_observation

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists(): raise ValueError('output must be a new directory')
    args.output.mkdir(parents=True)
    source=load_ct_case('parallel_2d/disk_analytic_32',data_root=Path(__file__).resolve().parents[1]/'test/data')
    records=[]
    for model in ('gaussian','poisson','poisson_gaussian'):
        y,public,meta=simulate_observation(source.measurement_clean,model=model,seed=17,sigma=.2,I0=100.,sigma_e=0. if model=='poisson' else 2.)
        case=replace(source,case_id='parallel_2d/p3_'+model,measurement=y,public_measurement=public,metadata={**source.metadata,'observation_domain':meta['observation_domain'],'measurement':meta})
        record=write_ct_case(case,args.output/model);record['path']=model;records.append(record)
    (args.output/'catalog.json').write_text(json.dumps({'schema_version':'1.0','cases':records},indent=2),encoding='utf-8')
if __name__=='__main__':main()

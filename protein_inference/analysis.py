import json,statistics
from protein_inference.config import path
from protein_inference.io import save

def run(c):
    records=[]
    for seed in c['sampling']['seeds']:
        p=path(c['output'])/f'seed-{seed}'/'EVALUATION-COMPLETE.json';r=json.loads(p.read_text())
        if not r['passed']:raise ValueError('Unfinished evaluation')
        records.append({'seed':seed,**r['metrics']})
    scalar={}
    for k,v in records[0].items():
        if k!='seed' and isinstance(v,(int,float)):
            vals=[r[k] for r in records];scalar[k]={'mean':statistics.mean(vals),'seed_sd':statistics.stdev(vals) if len(vals)>1 else None,'n_seeds':len(vals)}
    for subset in ('all','successful'):
        for metric in ('clusters','effective_clusters'):
            if 'diversity' in records[0]:
                vals=[r['diversity'][subset][metric] for r in records];scalar[f'{subset}_{metric}']={'mean':statistics.mean(vals),'seed_sd':statistics.stdev(vals) if len(vals)>1 else None,'n_seeds':len(vals)}
    save(path(c['output'])/'summary.json',{'config':c,'per_seed':records,'summary':scalar,'seed_sd_definition':'sample standard deviation ddof=1; undefined for one seed'})
    print(json.dumps(scalar,indent=2))

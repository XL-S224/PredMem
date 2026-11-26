"""Strict native backbone metrics; missing/non-finite outputs fail evaluation."""
import csv,json,math,os,subprocess,sys
from pathlib import Path
from PredMem.config import ROOT,path,load
from PredMem.io import save,sha

def run(c):
    for seed in c['sampling']['seeds']:evaluate_seed(c,path(c['output'])/f'seed-{seed}')

def evaluate_seed(c,out,worker=False):
    if not worker:
        config=out/'evaluation-resolved-config.json'
        save(config,c)
        cmd=[c['evaluation']['python'] or sys.executable,'-m','PredMem.evaluation','--resolved',str(config),'--seed-dir',str(out)]
        subprocess.run(cmd,check=True,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)})
        return
    if not (out/'GENERATION-COMPLETE.json').exists():raise ValueError('Generation has not completed')
    records=json.loads((out/'samples.json').read_text());task=c['task'];ev=c['evaluation']
    if not records:raise ValueError('No generated samples')
    if (out/'EVALUATION-COMPLETE.json').exists():raise FileExistsError('Evaluation already complete; preserve its receipt')
    if task!='structure_prediction':
        cmd=[ev['python'] or sys.executable,'-m','PredMem.refold','--manifest',str(out/'samples.json'),'--checkpoint',ev['refold_checkpoint']]
        with (out/'refold.log').open('w') as f:subprocess.run(cmd,check=True,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':str(ROOT)})
        refolds={(x['group'],x['id']):x for x in json.loads((out/'refolded/index.json').read_text())}
    import numpy as np
    from eval import get_pdb_file,get_pkl_file,cal_bb_rmsd,cal_bb_tm
    rows=[]
    motif_spec=load(path(c["dataset"]["motif_spans"])) if task=="motif" else None
    for r in records:
        if task in ('joint','motif'):left=Path(r['generated_pdb']);right=Path(refolds[r['group'],r['id']]['pdb'])
        elif task=='structure_prediction':left=Path(r['reference']);right=Path(r['generated_pdb'])
        else:left=Path(r['reference']);right=Path(refolds[r['group'],r['id']]['pdb'])
        ref_pkl=path(c['dataset']['reference_pickle_dir'])/(r['id']+'.pkl') if c['dataset']['reference_pickle_dir'] else None
        if task in ('structure_prediction','inverse_folding') and ref_pkl and ref_pkl.exists():a,seq,mask=get_pkl_file(ref_pkl)
        else:a,seq,mask=get_pdb_file(str(left))
        b,other,bmask=get_pdb_file(str(right));valid=mask[:,1]
        if len(a)!=len(b):raise ValueError('Reference and output residue count mismatch: '+r['id'])
        if not np.isfinite(a[valid]).all() or not np.isfinite(b[valid]).all():raise ValueError('Nonfinite coordinates')
        row={**r,'rmsd':float(cal_bb_rmsd(a[valid],b[valid])),'tm_score':float(cal_bb_tm(a[valid],b[valid],''.join(s for s,m in zip(seq,valid) if m)))}
        if task!='structure_prediction':row['plddt']=refolds[r['group'],r['id']]['plddt']
        if task=='inverse_folding':row['aar']=100*sum(x==y for x,y in zip(seq,r['sequence']))/len(seq)
        if task=='motif':
            group=r['group'];refname=motif_spec['variant_reference'][group]
            reference=path(c['dataset']['motif_dir'])/(refname+'_reference.pdb');xyz,_,_=get_pdb_file(str(reference))
            starts=motif_spec['reference_start'][refname];ends=motif_spec['reference_end'][refname];gt=np.concatenate([xyz[s:e+1] for s,e in zip(starts,ends)])
            spans=out/'start_end_scaffold'/(group+'.npz')
            if not spans.exists():spans=out/'native/start_end_scaffold'/(group+'.npz')
            with np.load(spans) as x:starts=x['start_idxs_list'][r['index']];ends=x['end_idxs_list'][r['index']]
            gen=np.concatenate([a[s:e+1] for s,e in zip(starts,ends)]);pred=np.concatenate([b[s:e+1] for s,e in zip(starts,ends)])
            row['gen_motif_rmsd']=float(cal_bb_rmsd(gen,gt));row['pred_motif_rmsd']=float(cal_bb_rmsd(pred,gt))
        row['success']=row['rmsd']<2 and (task!='motif' or row['gen_motif_rmsd']<1)
        for k in ('rmsd','tm_score','plddt','aar','gen_motif_rmsd'):
            if k in row and not math.isfinite(row[k]):raise ValueError('Invalid metric '+k)
        rows.append(row)
    save(out/'metrics.json',rows)
    fields=['group','id','length','rmsd','tm_score','plddt','aar','gen_motif_rmsd','pred_motif_rmsd','success']
    with (out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    summary={k:sum(r[k] for r in rows)/len(rows) for k in ('rmsd','tm_score','plddt','aar') if k in rows[0]}
    if task in ('joint','motif'):
        from PredMem.clustering import cluster
        summary['success_percent']=100*sum(r['success'] for r in rows)/len(rows)
        if task=='motif':summary['solved']=len({r['group'] for r in rows if r['success']})
        groups=sorted({r['group'] for r in rows});div={}
        for subset in ('all','successful'):
            vals=[cluster([Path(r['generated_pdb']) for r in rows if r['group']==g and (subset=='all' or r['success'])],out/'clusters'/subset/g,c) for g in groups]
            div[subset]={'clusters':sum(v['clusters'] for v in vals),'effective_clusters':sum(v['effective_clusters'] for v in vals),'by_group':dict(zip(groups,vals))}
        summary['diversity']=div
    save(out/'EVALUATION-COMPLETE.json',{'passed':True,'outputs':len(rows),'metrics':summary,'metrics_sha256':sha(out/'metrics.csv'),'criterion':'scRMSD < 2 A; motif additionally generated-motif RMSD < 1 A','geometry':'native evaluator: N/CA/C backbone RMSD and CA-only TM-align','evaluation_config':ev})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--resolved',required=True);p.add_argument('--seed-dir',required=True);a=p.parse_args()
    sys.path[:0]=[str(ROOT/'upstream/evals'),str(ROOT/'upstream/dplm/vendor/openfold')]
    evaluate_seed(json.loads(Path(a.resolved).read_text()),Path(a.seed_dir),worker=True)

"""Native ESM3 single-trajectory sampler with branch-isolated PM."""
import contextlib,copy,json,time,gzip
from pathlib import Path
from protein_inference.config import ROOT,path
from protein_inference.io import save,key,fasta,sha,safe_name

def run(c,seed,out):
    import torch,numpy as np,attr
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein,GenerationConfig,GenerationTrackConfig,MultiTrackGenerationConfig
    from esm.utils.structure.protein_chain import ProteinChain
    from esm.tokenization import get_invalid_tokenizer_ids
    from protein_inference.vendor.esm_engine import Engine,rng,clonebeam,clone
    from protein_inference.vendor.esm_feedback import SoftFeedback
    from protein_inference.schedules import factor
    from protein_inference.information import Observer
    torch.set_grad_enabled(False)
    model=ESM3.from_pretrained(c['model']['checkpoint']).to('cuda').eval()
    model.get_structure_decoder();model.get_function_decoder()
    measured={'calls':0,'items':0};native_forward=model.forward
    def counted(*a,**kw):
        tensors=[x for x in list(a)+list(kw.values()) if torch.is_tensor(x) and x.ndim>=2]
        measured['calls']+=1;measured['items']+=len(tensors[0])
        return native_forward(*a,**kw)
    model.forward=counted
    s=c['sampling'];d=c['decoding'];m=c['memory'];g=c['guidance'];task=c['task'];cases=[]
    if task=='joint':
        cases=[{'id':f'sample_{i}','case_key':f'full-benchmark-v1:main:L{L}:i{i}','group':f'length_{L}','index':i,'prompt':ESMProtein(sequence='_'*L)} for L in s['lengths'] for i in range(s['per_length'])]
    elif task=='structure_prediction':
        cases=[{'id':h,'group':c['dataset']['name'],'index':i,'prompt':ESMProtein(sequence=v),'reference':str(path(c['dataset']['reference_dir'])/(h+'.pdb'))} for i,(h,v) in enumerate(fasta(path(c['dataset']['fasta'])).items())]
    elif task=='inverse_folding':
        import pickle
        targets=fasta(path(c['dataset']['fasta']))
        for i,(h,sequence) in enumerate(targets.items()):
            p=path(c['dataset']['reference_dir'])/(h+'.pdb')
            pkl=path(c['dataset']['reference_pickle_dir'])/(h+'.pkl') if c['dataset']['reference_pickle_dir'] else None
            if pkl and pkl.exists():
                # Native benchmark conditioning: retain the continuous first-to-last valid-AA interval.
                with pkl.open('rb') as f:x=pickle.load(f)
                aa=np.asarray(x['aatype']);valid=np.flatnonzero(aa<20)
                if not len(valid):raise ValueError('No valid residues in '+h)
                xyz=np.asarray(x['atom_positions'])[valid[0]:valid[-1]+1]
                coords=torch.as_tensor(xyz,device='cuda')
            else:
                chain=ProteinChain.from_pdb(p,chain_id=c['dataset']['chain']);coords=torch.from_numpy(chain.atom37_positions).to('cuda')
            if len(coords)!=len(sequence):raise ValueError('Reference coordinate/sequence mismatch '+h)
            cases.append({'id':h,'group':c['dataset']['name'],'index':i,'prompt':ESMProtein(coordinates=coords),'reference':str(p),'reference_sequence':sequence})
    else:
        import sys
        sys.path.insert(0,str(ROOT/'upstream/esm'))
        from scaffolding_utils import motif_name_mapping,get_initial_scaffolding_batches
        (out/'start_end_scaffold').mkdir()
        for variant,reference in motif_name_mapping.items():
            if c['dataset']['motif_variants'] and variant not in c['dataset']['motif_variants']:continue
            p=path(c['dataset']['motif_dir'])/(reference+'_reference.pdb')
            with rng(key(seed,variant,'formal_batch_layout')):
                prompts,starts,ends,lens=get_initial_scaffolding_batches(str(p),variant,reference,s['per_motif'],'cpu')
            np.savez(out/'start_end_scaffold'/(variant+'.npz'),start_idxs_list=starts,end_idxs_list=ends,scaffold_length_list=lens)
            cases.extend({'id':f'{variant}_{i}','case_key':f'motif-formal:{variant}:{i}','group':variant,'index':i,'prompt':v,'reference':str(p)} for i,v in enumerate(prompts))
    if c['dataset']['limit'] is not None:cases=cases[:c['dataset']['limit']]
    records=[];total_calls=0
    for case in cases:
        case['group']=safe_name(case['group']);case['id']=safe_name(case['id']);prompt=case.pop('prompt')
        with rng(key(seed,case.get('case_key',case['id']),'encode')): initial=model.encode(prompt)
        length=len(initial)-2
        steps=max(1,(prompt.sequence.count('_') if task=='motif' else length)) if d['steps']=='length' else max(1,length//8) if d['steps']=='length/8' else d['steps']
        invalid=list(get_invalid_tokenizer_ids(model.tokenizers.sequence))
        def tc(tr,cls):
            return cls(num_steps=steps,temperature=d['temperature']['start'],temperature_annealing=d['temperature']['anneal'],allow_remask=d['remask'],strategy=d['unmasking'],invalid_ids=invalid if tr=='sequence' else [],cfg_scale=g[tr] if g['enabled'] else 0.,cfg_mode='drop_motif_cross_modal' if task=='motif' else 'cross_modal' if task=='joint' else 'drop_condition')
        if task in ('joint','motif'):
            conf=MultiTrackGenerationConfig(tracks={tr:tc(tr,GenerationTrackConfig) for tr in ('sequence','structure')},condition_on_coordinates_only=task=='motif')
        else:
            tr='structure' if task=='structure_prediction' else 'sequence';kw=attr.asdict(tc(tr,GenerationTrackConfig));conf=GenerationConfig(track=tr,**kw)
        eng=Engine(model,initial,conf,case['id']);state=clonebeam(eng.start);genkey=key(seed,case.get('case_key',case['id']),'generation');torch.cuda.synchronize();start=time.time();before=measured.copy();torch.cuda.reset_peak_memory_stats()
        hook=SoftFeedback(eng,mode='linear',lam=m['lambda'],k=m['k'],tracks=m['tracks'],summary_kind=m['summary']) if m['enabled'] else None
        info=out/'information'/case['group'];info.mkdir(parents=True,exist_ok=True)
        with rng(genkey),(hook if hook else contextlib.nullcontext()):
            obs=Observer(model,eng,info/(case['id']+'.json.gz'),m['k']) if c['diagnostics']['per_step'] else None
            with (obs if obs else contextlib.nullcontext()):
                for t in range(eng.T):
                    if obs: obs.state=state;obs.step=t
                    if hook:
                        hook.lam=m['lambda']*(factor(m['schedule'],t,eng.T) if m['schedule']!='static' else 1.)
                        state=hook.step(state,t)
                    else: state=eng.step(state,t)
        torch.cuda.synchronize();generation_seconds=time.time()-start;peak_bytes=torch.cuda.max_memory_allocated()
        actual_calls=measured['calls']-before['calls'];actual_items=measured['items']-before['items']
        assert actual_calls==eng.T
        if hook:
            assert hook.calls==eng.T and not sum(hook.trace[0]['replaced'].values())
            with gzip.open(info/(case['id']+'-memory.json.gz'),'wt') as f:json.dump(hook.trace,f)
        # Fixed sequence and coordinates are checked after native rollout.
        if initial.sequence is not None and state.protein_tensor.sequence is not None:
            fixed=initial.sequence!=model.tokenizers.sequence.mask_token_id
            assert torch.equal(initial.sequence[fixed],state.protein_tensor.sequence[fixed])
        if initial.coordinates is not None:
            assert torch.allclose(initial.coordinates,state.protein_tensor.coordinates,equal_nan=True)
        protein=model.decode(state.protein_tensor);p=out/'decoded_pdb'/case['group']/(case['id']+'.pdb');p.parent.mkdir(parents=True,exist_ok=True);protein.to_pdb(str(p))
        (out/(case['group']+'.fasta')).open('a').write('>'+case['id']+'\n'+protein.sequence+'\n')
        if c['diagnostics']['save_terminal_tokens']:torch.save({tr:getattr(state.protein_tensor,tr).detach().cpu() for tr in eng.tracks},info/(case['id']+'-terminal.pt'))
        total_calls+=eng.T
        row={**case,'sequence':protein.sequence,'length':len(protein.sequence),'generated_pdb':str(p),'pdb_sha256':sha(p),'seed':seed,'generation_key':genkey,'steps':eng.T,'forward_calls':actual_calls,'branch_items':actual_items,'generation_seconds':generation_seconds,'peak_allocated_bytes':peak_bytes,'seconds':time.time()-start}
        records.append(row);save(out/'samples.json',records)
    save(out/'counts.json',{'outputs':len(records),'denoiser_calls':total_calls,'sample_policy':'one terminal candidate per input; no best-of search'})

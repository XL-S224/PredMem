"""YAML-driven native DPLM entry points; cache and conditions stay branch-local."""
import contextlib,gzip,inspect,json,os,runpy,sys,time
from pathlib import Path
from protein_inference.config import ROOT,path
from protein_inference.io import save,sha,fasta

def run(c,seed,out):
    import torch
    from byprot.models.dplm2 import DPLM2Bit,MultimodalDiffusionProteinLanguageModel
    from protein_inference.vendor.dplm_pm import DPLMMemory,bit_code_logprob
    from protein_inference.vendor.remask_policy import RemaskPolicy
    from protein_inference.vendor.unused_hidden_state_suppression import UnusedHiddenStateSuppression
    d=c['decoding'];m=c['memory'];g=c['guidance'];s=c['sampling'];task=c['task'];bit=c['model']['family']=='dplm21'
    native=ROOT/'upstream/dplm';script=native/('guided_scaffold_generate_dplm2.py' if task=='motif' else 'guided_generate_dplm2.py')
    cls=DPLM2Bit if bit else MultimodalDiffusionProteinLanguageModel;original=cls.generate;sig=inspect.signature(original);batches=[]
    os.environ['DPLM_STATE_DIAGNOSTICS']='0'
    args=['--model_name',c['model']['checkpoint'],'--seed',str(seed),'--num_seqs',str(s['per_motif'] if task=='motif' else s['per_length']),'--batch_size',str(s['batch_size']),'--max_iter',str(d['steps'] if isinstance(d['steps'],int) else -1),'--saveto',str(out/'native'),'--sampling_strategy',f"annealing@{d['temperature']['start']}:{d['temperature']['end']}",'--unmasking_strategy',d['unmasking'],'--seq_cfg_scale',str(g['sequence'] if g['enabled'] else 0),'--struct_cfg_scale',str(g['structure'] if g['enabled'] else 0),'--cfg_mode','drop_motif_cross_modal' if task=='motif' else 'cross_modal' if task=='joint' else 'drop_condition']
    if bit:args+=['--bit_model']
    if task=='motif':args+=['--motif_aa',str(path(c['dataset']['motif_aa'])),'--motif_struct',str(path(c['dataset']['motif_structure']))]
    else:
        args+=['--task',{'joint':'co_generation','structure_prediction':'folding','inverse_folding':'inverse_folding'}[task],'--no_save_pdb' if task=='inverse_folding' else '--save_pdb','--seq_lens',*map(str,s['lengths'])]
        if task!='joint':
            fp=path(c['dataset']['fasta'])
            if c['dataset']['limit'] is not None:
                items=list(fasta(fp).items())[:c['dataset']['limit']];fp=out/'input.fasta';fp.write_text(''.join('>'+h+'\n'+seq+'\n' for h,seq in items))
            args+=['--input_fasta_path',str(fp)]
    if not d['temperature']['anneal']:
        args[args.index('--sampling_strategy')+1]='annealing@'+str(d['temperature']['start'])+':'+str(d['temperature']['start'])
    @torch.no_grad()
    def wrapped(self,*a,**kw):
        p=sig.bind(self,*a,**kw);p.apply_defaults();v=p.arguments
        length=v['input_tokens'].shape[1]//2-2
        v['max_iter']=max(1,length//8) if d['steps']=='length/8' else length if d['steps']=='length' else d['steps']
        T=v['max_iter'];start=time.time();cost={'calls':0,'items':0};rows=[];orig_forward=self.forward
        def counted(*aa,**kk):
            tensors=[x for x in list(aa)+list(kk.values()) if torch.is_tensor(x) and x.ndim>=2]
            ids=tensors[0];cost['calls']+=1;cost['items']+=len(ids);result=orig_forward(*aa,**kk)
            if c['diagnostics']['per_step']:
                info={};half=ids.shape[1]//2
                for tr in m['tracks']:
                    if tr=='sequence':z=(result['aatype_logits'] if bit else result['logits'][:,half:]).float()[...,4:24];mask=ids[:len(v['input_tokens']),half:]==self.aa_mask_id
                    elif bit:z=bit_code_logprob(result['struct_logits'].float().reshape(len(ids),half,13,2),pm.code_bits);mask=ids[:len(v['input_tokens']),:half]==self.struct_mask_id
                    else:z=result['logits'][:,:half].float()[...,36:36+pm.size];mask=ids[:len(v['input_tokens']),:half]==self.struct_mask_id
                    z=z[:len(mask)];lp=z.log_softmax(-1);q=lp.exp();h=-(q*lp).sum(-1)/__import__('math').log(2);track=[]
                    for prob,hh,mm in zip(q,h,mask):
                        mean=prob[mm].mean(0) if mm.any() else None
                        mi=float(-(mean*mean.clamp_min(1e-30).log2()).sum()-hh[mm].mean()) if mean is not None else None
                        top_ids=z[len(track)].topk(min(m['k'],z.shape[-1]),-1).indices+(4 if tr=='sequence' else 36)
                        track.append({'entropy_bits':hh.cpu().tolist(),'masked':mm.cpu().tolist(),'mutual_information_bits':mi,'topk_ids':top_ids.cpu().tolist()})
                    info[tr]=track
                rows.append({'step':cost['calls']-1,'tracks':info})
            return result
        self.forward=counted
        try:
            with (UnusedHiddenStateSuppression(self) if bit else contextlib.nullcontext()),RemaskPolicy(self,d['remask']) as policy:
                with DPLMMemory(self,v['input_tokens'],v.get('partial_masks'),lam=m['lambda'] if m['enabled'] else 0.,residual_lam=0.,k=m['k'],cfg_mode=v['cfg_mode'],enabled=m['tracks'],injection=m['interface'],schedule=m['schedule'],total_steps=T,summary_kind=m['summary']) as pm:
                    pm.diagnostic_steps=set()
                    result=original(*p.args,**p.kwargs)
            assert cost['calls']==T and pm.calls==T and pm.shadow_calls==0
            assert not sum(pm.trace[0]['replaced'].values())
            final=result['output_tokens'];variable=self.get_non_special_symbol_mask(v['input_tokens'],partial_masks=v.get('partial_masks'))
            assert torch.equal(final[~variable],v['input_tokens'][~variable])
            assert not (final.eq(self.aa_mask_id)|final.eq(self.struct_mask_id)).any()
            index=len(batches);record=out/'records';record.mkdir(exist_ok=True)
            with gzip.open(record/f'batch-{index:05d}-memory.json.gz','wt') as f:json.dump(pm.trace,f,allow_nan=False)
            if rows:
                with gzip.open(record/f'batch-{index:05d}-information.json.gz','wt') as f:json.dump(rows,f,allow_nan=False)
            if c['diagnostics']['save_terminal_tokens']:
                torch.save({'input':v['input_tokens'].cpu(),'partial_masks':v['partial_masks'].cpu() if v.get('partial_masks') is not None else None,'output':final.cpu()},record/f'batch-{index:05d}.pt')
            batches.append({'batch':index,'outputs':len(final),'length':length,'steps':T,'forward_calls':cost['calls'],'branch_items':cost['items'],'seconds':time.time()-start})
            save(out/'batches.json',batches)
            return result
        finally:self.forward=orig_forward
    cls.generate=wrapped;argv=sys.argv[:]
    try:
        sys.argv=[str(script)]+args
        with torch.no_grad():
            module=runpy.run_path(str(script),run_name='pm_native_entry')
            if task=='motif' and c['dataset']['motif_variants']:
                glob=inspect.unwrap(module['generate']).__globals__;glob['motif_name_mapping']={k:v for k,v in glob['motif_name_mapping'].items() if k in c['dataset']['motif_variants']}
            module['main']()
    finally:cls.generate=original;sys.argv=argv
    records=[]
    for f in sorted((out/'native').rglob('aatype.fasta')):
        group=f.parent.name
        for i,(h,seq) in enumerate(fasta(f).items()):
            pdb=f.parent/'decoded_pdb'/(h+'.pdb')
            row={'id':h,'index':i,'group':group,'sequence':seq,'length':len(seq),'seed':seed}
            if pdb.exists():row.update(generated_pdb=str(pdb),pdb_sha256=sha(pdb))
            if task in ('structure_prediction','inverse_folding'):
                row.update(reference=str(path(c['dataset']['reference_dir'])/(h+'.pdb')))
            records.append(row)
    if len(records)!=sum(x['outputs'] for x in batches):raise ValueError('Output manifest and generated sample count disagree')
    save(out/'samples.json',records);save(out/'counts.json',{'outputs':len(records),'denoiser_calls':sum(x['forward_calls'] for x in batches),'branch_items':sum(x['branch_items'] for x in batches),'native_command':args})

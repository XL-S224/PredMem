"""Audits native strict-cutoff remasking; never substitutes its result or changes RNG."""
import os
from pathlib import Path
import torch
class RemaskPolicy:
 def __init__(self,model,enabled):self.model=model;self.enabled=enabled;self.trace=[];self.exceptions=[]
 def __enter__(self):
  self.original=self.model._reparam_decoding;self.globals=self.original.__func__.__globals__;self.original_topk=self.globals['topk_masking'];self.last=[];self.step=None
  def topk(scores,cutoff_len,stochastic=False,temp=1.):
   cpu=torch.get_rng_state();device=scores.device.index if scores.is_cuda else None;gpu=torch.cuda.get_rng_state(device) if scores.is_cuda else None
   out=self.original_topk(scores,cutoff_len,stochastic=stochastic,temp=temp);count=out.sum(-1,keepdim=True);assert (count<=cutoff_len).all(),'native strict cutoff selected more than index budget'
   detail={'requested':cutoff_len.detach().cpu().flatten().tolist(),'actual':count.detach().cpu().flatten().tolist()}
   if (count<cutoff_len).any():
    cpu_after=torch.get_rng_state();gpu_after=torch.cuda.get_rng_state(device) if scores.is_cuda else None
    with torch.random.fork_rng(devices=[device] if scores.is_cuda else []):
     torch.set_rng_state(cpu)
     if gpu is not None:torch.cuda.set_rng_state(gpu,device)
     noise=-torch.log(-torch.log(torch.rand_like(scores)+1e-8)+1e-8) if stochastic else torch.zeros_like(scores)
     values=scores+temp*noise if stochastic else scores;assert not torch.isnan(values).any(),'nonfinite native ranking'
     cutoff=values.sort(-1)[0].gather(-1,cutoff_len);reconstructed=values<cutoff;assert torch.equal(reconstructed,out),'native topk mask replay differs'
    assert torch.equal(cpu_after,torch.get_rng_state())
    if gpu_after is not None:assert torch.equal(gpu_after,torch.cuda.get_rng_state(device))
    detail.update(strict_cutoff_replayed_exact=True,boundary_ties=(values==cutoff).sum(-1).cpu().tolist(),step=self.step,call=len(self.last),stochastic=stochastic,temp=temp)
    self.exceptions.append(detail)
    if os.environ.get('DPLM_MASK_AUDIT_DIR'):
     dest=Path(os.environ['DPLM_MASK_AUDIT_DIR']);dest.mkdir(parents=True,exist_ok=True);torch.save({'scores':scores.detach().cpu(),'noise':noise.detach().cpu(),'cutoff':cutoff.detach().cpu(),'mask':out.cpu(),'budget':cutoff_len.cpu(),'detail':detail},dest/f'boundary-{len(self.exceptions):05d}-step{self.step}.pt')
   self.last.append({'mask':out.clone(),'budget':cutoff_len.clone(),'detail':detail});return out
  def call(*args,**kw):
   assert not args;self.step=kw['t'];self.last=[];prev=kw['xt_neq_x0'].clone();variable=kw['non_special_sym_mask'];old=kw['output_tokens'].clone();scores=kw['cur_scores']
   if not self.enabled:kw=dict(kw);kw['cur_scores']=scores.masked_fill(variable&~prev,torch.inf)
   mask,tokens,sc=self.original(**kw)
   assert self.last,'expected actual native topk calls';union=torch.zeros_like(mask)
   for q in self.last:union|=q['mask']
   assert torch.equal(mask,union),'native transition mask differs from actual topk outputs'
   if not self.enabled:assert not (mask&~prev).any() and torch.equal(tokens[variable&~prev],old[variable&~prev])
   assert torch.equal(tokens[~variable],old[~variable]) and not (mask&~variable).any()
   counts={}
   for tr,kind in [('sequence',self.model.aa_type),('structure',self.model.struct_type)]:
    select=variable&kw['type_ids'].eq(kind);rate=1-kw['t']/kw['max_step'];expected=(select.sum(1).type_as(kw['output_scores'])*rate).long();actual=(mask&select).sum(1)
    assert (actual<=expected).all(),'native mask exceeds requested cutoff';counts[tr]={'remasked':int((mask&~prev&select).sum()),'newly_revealed':int((prev&~mask&select).sum()),'remaining':int(actual.sum()),'requested_per_row':expected.cpu().tolist(),'actual_per_row':actual.cpu().tolist()}
   self.trace.append({'step':kw['t'],'counts':counts,'native_topk_calls':[q['detail'] for q in self.last],'mask':mask.cpu().tolist()});return mask,tokens,sc
  self.globals['topk_masking']=topk;self.model._reparam_decoding=call;return self
 def __exit__(self,*a):self.model._reparam_decoding=self.original;self.globals['topk_masking']=self.original_topk

"""DPLM PM adapter, isolated from repository. Cache raw token maps; read after mask dropout.
DPLM2.1 top-k is over complete 13-bit codes, not over bit-head coordinates.
No training, additional model call, candidate, or random draw is introduced.
"""
import torch,os
from protein_inference.schedules import KINDS,factor
import torch.nn.functional as F

def bit_code_logprob(logits,code_bits):
 assert logits.shape[-1]==2 and logits.shape[-2]==code_bits.shape[-1]
 with torch.autocast(device_type=logits.device.type,enabled=False):
  lp=logits.float().log_softmax(-1)
  return lp[...,0].sum(-1,keepdim=True)+(lp[...,1]-lp[...,0])@code_bits.float().T

class DPLMMemory:
 def __init__(self,model,input_tokens,partial_masks=None,lam=.3,k=3,cfg_mode='cross_modal',enabled=('sequence','structure'),cache_mode='all_refresh',injection='pre',residual_lam=None,residual_mode='interpolate',schedule='static',total_steps=None,summary_kind='topk'):
  assert not model.training and input_tokens.ndim==2 and input_tokens.shape[0]>=1,'One independent trajectory per batch row'
  self.B=input_tokens.shape[0]
  assert input_tokens.shape[1]%2==0 and 0<=lam<=1 and k>=1
  assert cfg_mode in ('cross_modal','drop_condition','drop_motif','full_mask','drop_motif_cross_modal','motif_anchored_cross_modal')
  assert cache_mode in ('all_refresh','masked_last');self.cache_mode=cache_mode;self.cache_age={};self.topk={};self.snapshots=[]
  assert injection in ('pre','post_ln','pre_calibrated');self.injection=injection;self.norm_module=model.net.esm.encoder.layer[0].attention.LayerNorm;self.norm_handle=None;self.native_embedding=None
  assert schedule in KINDS
  assert schedule=='static' or (total_steps is not None and total_steps>=2 and injection=='post_ln' and residual_lam==0)
  self.schedule=schedule;self.total_steps=total_steps;self.base_lam=lam
  self.model=model;self.lam=lam;self.k=k;self.mode=cfg_mode;self.enabled=enabled
  self.residual_lam=lam if residual_lam is None else residual_lam
  assert 0<=self.residual_lam<=1 and residual_mode in ('interpolate','bounded_content')
  self.residual_mode=residual_mode;self.diagnostic_steps=set();self.custom_state_snapshots=[];self.guidance=None
  assert injection!='post_ln' or self.residual_lam==0 or residual_lam is None, 'post-LN mode has no residual intervention'
  self.active_memory=bool(self.lam or self.residual_lam)
  self.L=input_tokens.shape[1]//2
  self.variable=model.get_non_special_symbol_mask(input_tokens,partial_masks=partial_masks).detach().clone()
  self.summary_kind=summary_kind;assert summary_kind in ('topk','full_bit_mean','full_mean')
  self.bit=hasattr(model.net,'quant2emb');assert summary_kind!='full_bit_mean' or self.bit
  self.cache={};self.trace=[];self.calls=0;self.items=0;self.current=None;self.suspend=False;self.shadow_calls=0;self.state_snapshots=[];self.state_diagnostics=[]
  self.emb=model.net.esm.embeddings
  assert self.emb.position_embedding_type=='rotary' and self.emb.layer_norm is None,'Embedding-map derivation requires rotary/no embedding layernorm'
  self.word=self.emb.word_embeddings.weight
  self.aa_ids=torch.arange(4,24,device=input_tokens.device)
  self.size=model.struct_tokenizer.quantize.codebook_size
  self.code_ids=torch.arange(self.size,device=input_tokens.device)
  self.struct_ids=self.code_ids+36
  if self.bit:
   self.code_vectors=model.struct_tokenizer.quantize.get_codebook_entry(self.code_ids).float()
   self.code_bits=(self.code_vectors>0).float()
   weights=model.struct_tokenizer.quantize.mask.long()
   assert torch.equal((self.code_bits.long()*weights).sum(-1),self.code_ids),'bit ordering does not match actual native sampler'
   self.struct_table=F.linear(self.code_vectors,model.net.quant2emb.weight.float(),model.net.quant2emb.bias.float()).detach()
   if self.summary_kind in ('full_bit_mean','full_mean'):
    with torch.autocast(device_type=input_tokens.device.type,enabled=False):
     self.code_base=self.code_vectors[0];self.code_basis=self.code_vectors[weights]-self.code_base
     assert torch.allclose(self.code_base+self.code_bits@self.code_basis,self.code_vectors,atol=2e-6,rtol=0),'codebook is not affine in bits'
  else:self.struct_table=self.word[self.struct_ids].detach().float()
  self.raw_reference_rms={'sequence':self.word[self.aa_ids].detach().float().square().sum(-1).mean().sqrt(),'structure':self.struct_table.square().sum(-1).mean().sqrt()}
  self.ln_reference_rms={tr:F.layer_norm(table.float(),self.norm_module.normalized_shape,self.norm_module.weight.float(),self.norm_module.bias.float(),self.norm_module.eps).square().sum(-1).mean().sqrt() for tr,table in [('sequence',self.word[self.aa_ids]),('structure',self.struct_table)]}
 @staticmethod
 def schedule_factor(kind,step,total):
  return factor(kind,step,total)
 @staticmethod
 def residual_mix(native,memory,beta,kind,reference=None):
  assert 0<=beta<=1 and kind in ('interpolate','bounded_content')
  if beta==0:return native
  if kind=='interpolate':return ((1-beta)*native.float()+beta*memory.float()).to(native.dtype)
  assert reference is not None and torch.isfinite(reference).all() and (reference>0).all()
  ref=reference.float()[:,None,None];norm=memory.float().norm(dim=-1,keepdim=True)
  delta=beta*ref*memory.float()/torch.maximum(ref,norm).clamp_min(1e-12)
  assert torch.all(delta.norm(dim=-1)<=beta*ref.squeeze(-1)+2e-6)
  return (native.float()+delta).to(native.dtype)
 def capture_doses(self,norm_out,native_norm):
  t=self.calls-1
  if t not in self.diagnostic_steps:return
  ids=self.current;s2=self._scale(ids);s1=self._scale(ids[:,self.L:]) if self.bit else s2
  result={}
  for tr in self.enabled:
   allow=self._permitted(ids,tr)[:self.B];offset=self.L if tr=='sequence' else 0;scale=s2*s1 if tr=='sequence' or not self.bit else s2
   actual=self.norm_input[:self.B,offset:offset+self.L].float();base=self.native_embedding[:self.B,offset:offset+self.L].float();u=norm_out[:self.B,offset:offset+self.L].float();u0=native_norm[:self.B,offset:offset+self.L].float();raw_delta=(actual-base).norm(dim=-1);norm_delta=(u-u0).norm(dim=-1);reference=scale[:self.B]*self.raw_reference_rms[tr]
   result[tr]=[]
   for i in range(self.B):
    mask=allow[i];n=int(mask.sum());result[tr].append({'index':i,'masked_positions':n,'raw_delta_l2_mean':raw_delta[i][mask].mean().item() if n else None,'raw_relative_RMS':(raw_delta[i][mask].square().mean().sqrt()/reference[i]).item() if n else None,'LN_delta_l2_mean':norm_delta[i][mask].mean().item() if n else None,'LN_relative_RMS_unscaled_legal_reference':(norm_delta[i][mask].square().mean().sqrt()/self.ln_reference_rms[tr]).item() if n else None,'raw_legal_reference_rms':reference[i].item(),'native_scaling':scale[i].item()})
  self._audit['dose_per_trajectory']=result
  if self.active_memory:
   self.custom_state_snapshots.append({'step':t,'input_tokens':ids[:1].detach().cpu(),'native_embedding':self.native_embedding[:1].detach().cpu(),'residual_embedding':self.norm_input[:1].detach().cpu(),'first_LN_output':norm_out[:1].detach().cpu(),'cache_before':{tr:v[:1].detach().cpu() for tr,v in self.cache.items()},'scope':'actual primary branch trajectory0, before current cache refresh'})
 def capture_distributions(self,out):
  if self.calls-1 not in self.diagnostic_steps:return
  result={}
  for tr in self.enabled:
   if tr=='sequence':raw=out['aatype_logits'] if self.bit else out['logits'][:,self.L:];logits=raw.float().index_select(-1,self.aa_ids);drop=1
   elif self.bit:raw=out['struct_logits'].float().reshape(len(self.current),self.L,-1,2);logits=bit_code_logprob(raw,self.code_bits);drop=2
   else:logits=out['logits'][:,:self.L].float().index_select(-1,self.struct_ids);drop=2
   allow=self._permitted(self.current,tr)[:self.B];guided=logits[:self.B]
   if self.guidance is not None:uncond=logits[drop*self.B:(drop+1)*self.B];guided=uncond+self.guidance[tr]*(guided-uncond)
   result[tr]={}
   for distribution,z in [('conditional',logits[:self.B]),('CFG',guided)]:
    logp=z.log_softmax(-1);p=logp.exp();entropy=-(p*logp).sum(-1);mass=p.topk(self.k,-1).values.sum(-1);confidence=p.max(-1).values
    rows=[]
    for i in range(self.B):
     mask=allow[i];n=int(mask.sum());rows.append({'index':i,'masked_positions':n,'entropy_nats':entropy[i][mask].mean().item() if n else None,'topk_mass':mass[i][mask].mean().item() if n else None,'max_probability':confidence[i][mask].mean().item() if n else None,'temperature':1.0})
    result[tr][distribution]=rows
  self._audit['distribution_per_trajectory']=result
 def _scale(self,ids):
  if not self.emb.token_dropout:return torch.ones(ids.shape[0],device=ids.device)
  mask=ids.eq(self.emb.mask_token_id);length=ids.ne(self.emb.padding_idx).sum(-1)
  return .88/(1-mask.sum(-1).float()/length)
 def _permitted(self,ids,track):
  B=ids.shape[0];branches=B//self.B;assert B==branches*self.B and branches in (1,2,3),'native or CFG batch layout required'
  branch_rows=[0] if branches==1 else ([0,1] if track=='sequence' else ([0,2] if branches==3 else [0,1]))
  rows=[b*self.B+i for b in branch_rows for i in range(self.B)]
  local=ids[:,self.L:] if track=='sequence' else ids[:,:self.L]
  variable=self.variable[:,self.L:] if track=='sequence' else self.variable[:,:self.L]
  allowed=variable.repeat(branches,1)&local.eq(self.model.aa_mask_id if track=='sequence' else self.model.struct_mask_id)
  rowmask=torch.zeros(B,1,dtype=torch.bool,device=ids.device);rowmask[rows]=True
  return allowed&rowmask
 def __enter__(self):
  self.original=self.model.forward
  def hook(module,args,kwargs,output):
   if kwargs.get('inputs_embeds') is None or self.suspend:return output
   assert self.current is not None
   ids=kwargs['input_ids'];assert torch.equal(ids,self.current)
   ans=output
   self._read_count+=1
   self._audit['post_dropout_reads']+=1
   self.native_embedding=output
   if self.injection=='post_ln':return output
   if not self.cache or self.residual_lam==0:return ans
   ans=output.clone();s2=self._scale(ids)
   s1=self._scale(ids[:,self.L:]) if self.bit else s2
   for tr in self.enabled:
    if tr not in self.cache:continue
    allow=self._permitted(ids,tr);offset=self.L if tr=='sequence' else 0
    scale=s2*s1 if tr=='sequence' or not self.bit else s2
    raw=self.cache[tr];assert raw.shape[:2]==allow.shape
    mix=raw*scale[:,None,None]
    before=ans[:,offset:offset+self.L].clone();after=before.clone()
    reference=scale*self.raw_reference_rms[tr];mixed=self.residual_mix(before,mix,self.residual_lam,self.residual_mode,reference);after[allow]=mixed[allow]
    assert torch.equal(after[~allow],before[~allow]) and torch.isfinite(after).all()
    ans[:,offset:offset+self.L]=after
    self._audit['replaced'][tr]=int(allow.sum())
    self._audit['read_scaling'][tr]=scale.cpu().tolist()
    self._audit['read_positions'][tr]=allow.nonzero().cpu().tolist()
    assert (self.cache_age[tr][allow]>=0).all(),'reading invalid cache'
    age=self.calls-1-self.cache_age[tr]
    self._audit.setdefault('cache_read_age',{})[tr]={'positions':int(allow.sum()),'age_gt_one':int((allow&(age>1)).sum()),'max_age':int(age[allow].max()) if allow.any() else None}
    if allow.any():
     self._audit['norms'][tr]={'base_mean':before[allow].float().norm(dim=-1).mean().item(),'memory_mean':mix[allow].norm(dim=-1).mean().item(),'mixed_mean':after[allow].float().norm(dim=-1).mean().item(),'actual_delta_mean':(after[allow].float()-before[allow].float()).norm(dim=-1).mean().item(),'native_track_energy':before.float().square().sum(-1).sum(-1).cpu().tolist(),'feedback_track_energy':after.float().square().sum(-1).sum(-1).cpu().tolist()}
   return ans
  self.handle=self.emb.register_forward_hook(hook,with_kwargs=True)
  def norm_hook(module,args,output):
   if self.suspend:return output
   self._norm_read_count+=1;self.norm_input=args[0].detach();self.norm_output=output.detach()
   native=F.layer_norm(self.native_embedding.float(),module.normalized_shape,module.weight.float(),module.bias.float(),module.eps)
   if self.injection=='pre' or not self.cache or not self.active_memory or (self.injection=='post_ln' and self.lam==0):
    self._audit['first_norm_reads']=self._norm_read_count
    for tr in self.enabled:
     allow=self._permitted(self.current,tr);offset=self.L if tr=='sequence' else 0
     delta=(output[:,offset:offset+self.L].float()-native[:,offset:offset+self.L])[allow]
     self._audit.setdefault('normalized_delta',{})[tr]=float(delta.norm(dim=-1).mean()) if allow.any() else None
    self.capture_doses(output,native)
    return output
   ids=self.current;ans=output.clone();s2=self._scale(ids);s1=self._scale(ids[:,self.L:]) if self.bit else s2
   for tr in self.enabled:
    if tr not in self.cache:continue
    allow=self._permitted(ids,tr);offset=self.L if tr=='sequence' else 0;scale=s2*s1 if tr=='sequence' or not self.bit else s2
    raw=self.cache[tr]*scale[:,None,None]
    target=F.layer_norm(raw.float(),module.normalized_shape,module.weight.float(),module.bias.float(),module.eps)
    actual_before=output[:,offset:offset+self.L];before=native[:,offset:offset+self.L] if self.injection=='pre_calibrated' else actual_before;after=actual_before.clone();after[allow]=((1-self.lam)*before[allow].float()+self.lam*target[allow]).to(after.dtype)
    assert torch.equal(after[~allow],actual_before[~allow]) and torch.isfinite(after).all()
    expected=self.lam*(target[allow]-before[allow].float());observed=after[allow].float()-before[allow].float();assert torch.allclose(observed,expected,atol=2e-6,rtol=1e-4),'post-LN dose not linear'
    ans[:,offset:offset+self.L]=after;self._audit['replaced'][tr]=int(allow.sum());self._audit['read_positions'][tr]=allow.nonzero().cpu().tolist();self._audit['read_scaling'][tr]=scale.cpu().tolist()
    age=self.calls-1-self.cache_age[tr];self._audit.setdefault('cache_read_age',{})[tr]={'positions':int(allow.sum()),'age_gt_one':int((allow&(age>1)).sum()),'max_age':int(age[allow].max()) if allow.any() else None}
    if allow.any():self._audit['norms'][tr]={'base_mean':before[allow].float().norm(dim=-1).mean().item(),'memory_mean':target[allow].norm(dim=-1).mean().item(),'mixed_mean':after[allow].float().norm(dim=-1).mean().item(),'actual_delta_mean':observed.norm(dim=-1).mean().item(),'space':'after first attention LayerNorm'}
   self.capture_doses(ans,native);self.norm_output=ans.detach();return ans
  self.norm_handle=self.norm_module.register_forward_hook(norm_hook)
  def forward(*args,**kwargs):
   ids=kwargs.get('input_ids',args[0] if args else None);assert ids.shape[1]==2*self.L
   self.lam=self.base_lam*self.schedule_factor(self.schedule,self.calls,self.total_steps)
   self.current=ids;self.calls+=1;self.items+=len(ids);self._read_count=0;self._norm_read_count=0
   self._audit={'summary_kind':self.summary_kind,'diagnostic_k':self.k,'schedule':self.schedule,'effective_lambda':self.lam,'step':self.calls-1,'branch_items':len(ids),'post_dropout_reads':0,'replaced':{},'read_scaling':{},'read_positions':{},'norms':{},'retained_mass':{}}
   cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state(ids.device) if ids.is_cuda else None
   out=self.original(*args,**kwargs)
   assert self._read_count==1 and self._norm_read_count==1,'one native embedding and one first-LN hook expected'
   assert torch.equal(cpu,torch.get_rng_state())
   if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state(ids.device))
   if os.environ.get('DPLM_STATE_DIAGNOSTICS','0')=='1' and self.calls-1 in (0,1,4,24,124,249,374,499):
    if self.active_memory and self.cache:
     self.suspend=True
     try:shadow=self.original(*args,**kwargs);self.shadow_calls+=1
     finally:self.suspend=False
    else:shadow=out
    assert torch.equal(cpu,torch.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state(ids.device))
    chosen=torch.tensor([0,self.B,2*self.B],device=ids.device)
    snapshot={'step':self.calls-1,'input_tokens':ids.index_select(0,chosen).cpu(),'original_variable':self.variable[:1].cpu(),'lambda':self.lam,'injection':self.injection,'first_norm_input':self.norm_input.index_select(0,chosen).cpu(),'first_norm_output':self.norm_output.index_select(0,chosen).cpu(),'native_embedding':self.native_embedding.index_select(0,chosen).detach().cpu(),'cache_before':{tr:v.index_select(0,chosen).cpu() for tr,v in self.cache.items()},'cache_step_before':{tr:v.index_select(0,chosen).cpu() for tr,v in self.cache_age.items()},'selected_batch_rows':chosen.cpu(),'main_logits':{k:v.index_select(0,chosen).detach().cpu() for k,v in out.items() if 'logits' in k and torch.is_tensor(v)},'shadow_logits':{k:v.index_select(0,chosen).detach().cpu() for k,v in shadow.items() if 'logits' in k and torch.is_tensor(v)}}
    self.state_snapshots.append(snapshot)
    for tr in ('sequence','structure'):
     mask=self._permitted(ids,tr)[:self.B]
     def score(output,guided):
      if tr=='sequence':raw=output['aatype_logits'] if self.bit else output['logits'][:,self.L:];raw=raw.detach().float().index_select(-1,self.aa_ids)
      elif self.bit:raw=output['struct_logits'].detach().float().reshape(len(ids),self.L,13,2)
      else:raw=output['logits'][:,:self.L].detach().float().index_select(-1,self.struct_ids)
      c,us,ux=raw.chunk(3,0);u=us if tr=='sequence' else ux
      raw=u+(2. if tr=='sequence' else 1.5)*(c-u) if guided else c
      return bit_code_logprob(raw,self.code_bits) if tr=='structure' and self.bit else raw
     for guided in (False,True):
      lp=score(out,guided).log_softmax(-1);lq=score(shadow,guided).log_softmax(-1);prob=lp.exp();reference=lq.exp()
      kl=(prob*(lp-lq)).sum(-1);flip=lp.argmax(-1)!=lq.argmax(-1);hp=-(prob*lp).sum(-1);hq=-(reference*lq).sum(-1)
      assert torch.isfinite(kl).all() and kl.min()>-1e-4
      v={'step':self.calls-1,'track':tr,'distribution':'CFG' if guided else 'conditional','mask_count':int(mask.sum()),'KL_memory_vs_native':float(kl[mask].mean()) if mask.any() else None,'top1_change_fraction':float(flip[mask].float().mean()) if mask.any() else None,'entropy_memory':float(hp[mask].mean()) if mask.any() else None,'entropy_native':float(hq[mask].mean()) if mask.any() else None,'per_trajectory':[{'index':i,'mask_count':int(mask[i].sum()),'KL':float(kl[i][mask[i]].mean()) if mask[i].any() else None,'top1_change_fraction':float(flip[i][mask[i]].float().mean()) if mask[i].any() else None} for i in range(self.B)]}
      self.state_diagnostics.append(v)
    if shadow is not out:del shadow
   if self.active_memory or self.calls-1 in self.diagnostic_steps:
    cache={}
    for tr in self.enabled:
     if tr=='sequence':
      raw=out['aatype_logits'] if self.bit else out['logits'][:,self.L:]
      logits=raw.detach().float().index_select(-1,self.aa_ids);table=self.word[self.aa_ids].detach().float()
     elif self.bit:
      raw=out['struct_logits'].detach().float().reshape(len(ids),self.L,-1,2)
      logits=bit_code_logprob(raw,self.code_bits);table=self.struct_table
     else:logits=out['logits'][:,:self.L].detach().float().index_select(-1,self.struct_ids);table=self.struct_table
     values,index=logits.topk(self.k,-1);p=values.softmax(-1)
     candidate=(p[...,None]*table[index]).sum(-2).detach()
     if self.summary_kind in ('full_bit_mean','full_mean'):
      with torch.autocast(device_type=logits.device.type,enabled=False):
       if tr=='sequence' or not self.bit:candidate=logits.float().softmax(-1)@table.float()
       else:
        bit_p1=raw.float().softmax(-1)[...,1];expected_code=self.code_base+bit_p1@self.code_basis;candidate=F.linear(expected_code,self.model.net.quant2emb.weight.float(),self.model.net.quant2emb.bias.float())
      candidate=candidate.detach()
     eligible=self._permitted(ids,tr)
     update=torch.ones_like(eligible) if self.cache_mode=='all_refresh' else eligible
     old=self.cache.get(tr,torch.zeros_like(candidate))
     previous_age=self.cache_age.get(tr,torch.full_like(eligible,-1,dtype=torch.long))
     if self.calls==1:assert (update|~eligible).all()
     cache[tr]=torch.where(update[...,None],candidate,old).detach()
     assert torch.equal(cache[tr][~update],old[~update]),'frozen cache was overwritten'
     self.cache_age[tr]=torch.where(update,torch.full_like(previous_age,self.calls-1),previous_age)
     previous=self.topk.get(tr,{'ids':torch.zeros_like(index),'weights':torch.zeros_like(p)})
     ids_table=self.aa_ids if tr=='sequence' else self.struct_ids
     self.topk[tr]={'ids':torch.where(update[...,None],ids_table[index],previous['ids']).detach(),'weights':torch.where(update[...,None],p,previous['weights']).detach()}
     assert torch.isfinite(cache[tr]).all()
     self._audit.setdefault('cache_updates',{})[tr]={'updated':int(update.sum()),'frozen':int((~update).sum()),'masked_eligible':int(eligible.sum())}
     mass=(values.logsumexp(-1)-logits.logsumexp(-1)).exp()
     self._audit['retained_mass'][tr]={'eligible_count':int(eligible.sum()),'mean':mass[eligible].mean().item() if eligible.any() else None}
     self._audit.setdefault('actual_cache_mass',{})[tr]={'eligible_count':int(eligible.sum()),'mean':(1.0 if self.summary_kind!='topk' else mass[eligible].mean().item()) if eligible.any() else None,'topk_mass_is_diagnostic_only':self.summary_kind!='topk'}
    self.cache=cache
    self.capture_distributions(out)
    if self.calls in (1,125,250,375,500):self.snapshots.append({'summary_kind':self.summary_kind,'topk_role':'diagnostic_only' if self.summary_kind in ('full_bit_mean','full_mean') else 'actual_cache_weights','step':self.calls-1,'input_tokens':ids.detach().cpu(),'topk':{tr:{k:v.cpu() for k,v in item.items()} for tr,item in self.topk.items()},'cache_step':{tr:v.cpu() for tr,v in self.cache_age.items()}})
   self.trace.append(self._audit);self.current=None
   return out
  self.model.forward=forward
  return self
 def __exit__(self,*args):
  self.model.forward=self.original;self.handle.remove();self.norm_handle.remove();self.current=None;self.cache={}

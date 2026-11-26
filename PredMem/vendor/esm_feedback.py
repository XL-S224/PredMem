"""One hard trajectory with branch-separated predictive embedding feedback."""
import torch
def construct(mask, mix, mode, lam):
 if lam==0:return mask.expand_as(mix)
 if mode=='linear':return (1-lam)*mask+lam*mix
 a=mask/mask.norm().clamp_min(1e-12);b=mix/mix.norm(dim=-1,keepdim=True).clamp_min(1e-12)
 theta=torch.acos((a*b).sum(-1,keepdim=True).clamp(-1+1e-7,1-1e-7))
 return mask.norm()*(torch.sin((1-lam)*theta)*a+torch.sin(lam*theta)*b)/torch.sin(theta)
def replace(ids, output, cache, mask_id, positions, rows, mode, lam, weight, source_masks=None):
 if cache is None or lam==0:return output,0
 assert ids.shape[0] in (1,2,3) and ids.shape==output.shape[:2]
 eligible=torch.zeros_like(ids,dtype=torch.bool);eligible[:,positions]=True
 allow=torch.zeros_like(ids,dtype=torch.bool)
 for row in rows:
  allow[row]=eligible[row]&(ids[row]==mask_id)
  if source_masks is not None:allow[row]&=source_masks[row]
 result=output.clone();n=0
 for row in rows:
  dest=allow[row].nonzero().flatten()
  if not len(dest):continue
  mix=cache[row][dest]
  if mode=='shuffled' and len(dest)>1:mix=mix.roll(len(dest)//2,0)
  value=construct(weight[mask_id].float(),mix.float(),'linear' if mode=='linear' else 'spherical',lam)
  assert torch.isfinite(value).all()
  result[row,dest]=value.to(result.dtype);n+=len(dest)
 assert torch.equal(result[~allow],output[~allow]),'changed hard/fixed/dropped branch'
 return result,n
class SoftFeedback:
 def __init__(self,engine,mode='linear',lam=.3,k=3,tracks=None,source_policy='all',summary_kind='topk'):
  self.eng=engine;self.model=engine.model;self.mode=mode;self.lam=lam;self.k=k
  assert summary_kind in ('topk','full_mean');self.summary_kind=summary_kind
  assert mode in ['spherical','linear','shuffled','zero']
  self.enabled=tuple(engine.tracks if tracks is None else tracks);assert self.enabled and len(set(self.enabled))==len(self.enabled) and set(self.enabled)<=set(['sequence','structure'])
  assert source_policy in ['all','mask_only']
  self.source_policy=source_policy;self.cache_source={};self.cache={};self.trace=[];self.handles=[];self.calls=0;self.items=0;self.active=False
  assert set(self.enabled)<=set(engine.tracks)
  guided = any((engine.config.get_track(tr) if engine.multi else engine.config).cfg_scale > 0 for tr in engine.tracks)
  self.nbranches=3 if guided and engine.multi else 2 if guided else 1
  self.rows={tr:([0] if not guided else [0,1] if tr=='sequence' or not engine.multi else [0,2]) for tr in engine.tracks}
 def __enter__(self):
  self.original=self.model.forward
  self.modules={'sequence':self.model.encoder.sequence_embed,'structure':self.model.encoder.structure_tokens_embed}
  self.modules={tr:module for tr,module in self.modules.items() if tr in self.enabled}
  for tr,module in self.modules.items():
   def hook(mod,args,out,tr=tr):
    before=torch.get_rng_state();gpu=torch.cuda.get_rng_state() if out.is_cuda else None
    mask_id=getattr(self.eng.tok,tr).mask_token_id
    self.current_source[tr]={row:(args[0][row]==mask_id).clone() for row in self.rows[tr]}
    if tr in self.cache:
     for row in self.rows[tr]:
      dest=torch.zeros_like(args[0][row],dtype=torch.bool);dest[self.eng.positions[tr]]=True;dest&=args[0][row]==mask_id
      trusted=dest&self.cache_source[tr][row]
      self.audit['source_mask_eligible'][tr]+=int(trusted.sum())
      self.audit['source_visible_eligible'][tr]+=int((dest&~self.cache_source[tr][row]).sum())
    ans,n=replace(args[0],out,self.cache.get(tr),getattr(self.eng.tok,tr).mask_token_id,self.eng.positions[tr],self.rows[tr],self.mode,0 if self.mode=='zero' else self.lam,mod.weight,self.cache_source.get(tr) if self.source_policy=='mask_only' else None)
    if n:
     changed=(ans-out).float().norm(dim=-1)
     changed_mask=torch.zeros_like(args[0],dtype=torch.bool)
     for rr in self.rows[tr]:
      changed_mask[rr,self.eng.positions[tr]]=args[0][rr,self.eng.positions[tr]]==mask_id
     self.audit.setdefault('norms',{})[tr]={'space':'token embedding','positions':n,'actual_delta_l2_mean':changed[changed_mask].mean().item(),'relative_delta_rms':(changed[changed_mask].square().mean().sqrt()/mod.weight.detach().float()[self.eng.valid[tr]].square().sum(-1).mean().sqrt()).item()}
    assert torch.equal(before,torch.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    self.audit['replaced'][tr]=n
    self.audit['filtered'][tr]=self.audit['source_visible_eligible'][tr] if self.source_policy=='mask_only' and tr in self.cache else 0
    expected=self.audit['source_mask_eligible'][tr]+(self.audit['source_visible_eligible'][tr] if self.source_policy=='all' else 0)
    assert n==(expected if self.lam and self.mode!='zero' else 0)
    return ans
   self.handles.append(module.register_forward_hook(hook))
  def forward(*args,**kw):
   assert self.active,'feedback outside generation'
   self.calls+=1
   ts=[x for x in list(args)+list(kw.values()) if torch.is_tensor(x)];assert ts and ts[0].shape[0]==self.nbranches
   self.items+=self.nbranches
   self.current_source={}
   out=self.original(*args,**kw)
   # Store only each branch's permitted modality. No conditional cache is shared.
   new={}
   for tr,module in self.modules.items():
    logits=getattr(out,tr+'_logits').detach().float();assert logits.shape[0]==self.nbranches
    ids=torch.tensor(self.eng.valid[tr],device=logits.device)
    local=logits.index_select(-1,ids)
    v,idx=local.topk(self.k,dim=-1);p=v.softmax(-1);emb=module.weight[ids[idx]].detach().float()
    mean=(p[...,None]*emb).sum(-2)
    if self.summary_kind=='full_mean':mean=local.softmax(-1)@module.weight[ids].detach().float()
    new[tr]={row:mean[row].clone() for row in self.rows[tr]}
    assert all(torch.isfinite(x).all() for x in new[tr].values())
   self.cache=new;self.cache_source=self.current_source
   return out
  self.model.forward=forward;return self
 def step(self,state,t):
  self.audit=dict(step=t,source_policy=self.source_policy,source_mask_eligible={tr:0 for tr in self.eng.tracks},source_visible_eligible={tr:0 for tr in self.eng.tracks},filtered={tr:0 for tr in self.eng.tracks},replaced={tr:0 for tr in self.eng.tracks},enabled_tracks=list(self.enabled),branch_isolated=True,hard_fixed_dropped_unchanged=True,hook_rng_preserved=True)
  before=self.calls;self.active=True
  try:result=self.eng.step(state,t)
  finally:self.active=False
  assert self.calls==before+1
  if t==0:assert sum(self.audit['replaced'].values())==0
  self.trace.append(self.audit);return result
 def __exit__(self,*args):
  self.model.forward=self.original
  for h in self.handles:h.remove()

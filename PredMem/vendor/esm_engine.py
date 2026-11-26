"""Strict Single adapter to unchanged author step kernels; no scoring or search API."""

import math,random,copy,contextlib,time
import numpy as np
import torch,attr
from esm.sdk.api import ESMProtein,ESMProteinTensor,GenerationConfig,MultiTrackGenerationConfig,GenerationTrackConfig
from esm.sdk.experimental import beam_search as b
from esm.tokenization import get_invalid_tokenizer_ids

def clone(pt):
 out=attr.evolve(pt)
 for f in attr.fields(ESMProteinTensor):
  x=getattr(pt,f.name)
  if torch.is_tensor(x):setattr(out,f.name,x.detach().clone())
 return out

def clonebeam(state):return b._BeamState(clone(state.protein_tensor),copy.deepcopy(state.entropy),copy.deepcopy(state.initial_mask))

@contextlib.contextmanager
def rng(seed):
 py=random.getstate();npstate=np.random.get_state()
 with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  try:yield
  finally:random.setstate(py);np.random.set_state(npstate)

class Engine:
 def __init__(self,model,initial,config,case_id):
  self.model=model;self.tok=model.tokenizers;self.config=copy.deepcopy(config);self.original=clone(initial);self.case_id=case_id
  self.multi=isinstance(config,MultiTrackGenerationConfig);self.tracks=config.get_all_tracks() if self.multi else [config.track]
  self.total={};self.positions={};self.valid={};self.cache={};self.calls=0;self.batch_items=0
  pt=clone(initial)
  if config.condition_on_coordinates_only and pt.coordinates is not None:pt.structure=None
  filled=b._stack_protein_tensors([pt],[len(pt)],self.tok,pt.device);p=filled.slice(0,sequence_len=len(pt));p.coordinates=pt.coordinates
  masks={}
  for tr in self.tracks:
   tc=self.config.get_track(tr) if self.multi else self.config
   x=getattr(filled,tr);m=b._get_masked_positions(tr,x[0:1],getattr(self.tok,tr).mask_token_id)
   self.total[tr]=int(m.sum());masks[tr]=m if tc.allow_remask else None
   if 0<self.total[tr]<tc.num_steps:tc.num_steps=self.total[tr]
   score=m[0].clone()
   if tr=='structure' and self.multi and initial.coordinates is not None and config.condition_on_coordinates_only:
    fixed=torch.isfinite(initial.coordinates).any(-1).any(-1)
    if initial.sequence is not None:fixed|=initial.sequence!=self.tok.sequence.mask_token_id
    score=~fixed
   score[0]=False;score[-1]=False;self.positions[tr]=score.nonzero().flatten()
   self.valid[tr]=sorted(set(getattr(self.tok,tr).all_token_ids)-set(getattr(self.tok,tr).special_token_ids)-set(tc.invalid_ids))
  self.T=self.config.get_max_num_steps() if self.multi else self.config.num_steps
  self.start=b._BeamState(p,None,masks if self.multi else masks[self.tracks[0]])
  self.N=sum(len(i) for i in self.positions.values());assert self.N>0

 def step(self,state,t,seed=None):
  fn=b._multi_track_sample_step if self.multi else b._single_track_sample_step
  def perform():return fn(clonebeam(state),t,self.config,self.model,self.tok,self.total if self.multi else self.total[self.tracks[0]],self.original)
  if seed is None:return perform()
  with rng(seed):return perform()

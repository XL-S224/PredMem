"""Suppress unused intermediate-state collection, preserving native batch/precision/math."""
class UnusedHiddenStateSuppression:
 def __init__(self,model):self.model=model;self.calls=0;self.requested=[]
 def __enter__(self):
  def pre(module,args,kwargs):
   self.calls+=1;self.requested.append(kwargs.get('output_hidden_states'));assert kwargs.get('output_hidden_states') is True
   kwargs=dict(kwargs);kwargs['output_hidden_states']=False;return args,kwargs
  def post(module,args,kwargs,result):
   assert 'all_hidden_states' not in result
   result['all_hidden_states']=None;return result
  self.a=self.model.net.register_forward_pre_hook(pre,with_kwargs=True);self.b=self.model.net.register_forward_hook(post,with_kwargs=True);return self
 def __exit__(self,*args):self.a.remove();self.b.remove()

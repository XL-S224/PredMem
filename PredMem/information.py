"""Per-step conditional predictive statistics; entropy and I(J;X) are in bits."""
import math
class Observer:
    def __init__(self,model,engine,out,topk=3):self.model=model;self.engine=engine;self.out=out;self.k=topk;self.rows=[];self.state=None;self.step=0
    def __enter__(self):
        import torch
        self.original=self.model.forward
        def forward(*a,**kw):
            result=self.original(*a,**kw); e=self.engine;row={'step':self.step,'tracks':{}}
            for tr in e.tracks:
                pos=e.positions[tr];valid=torch.tensor(e.valid[tr],device=pos.device)
                logits=getattr(result,tr+'_logits')[0].index_select(0,pos).index_select(-1,valid).float()
                lp=logits.log_softmax(-1);prob=lp.exp();h=-(prob*lp).sum(-1)/math.log(2)
                mask=getattr(self.state.protein_tensor,tr).index_select(0,pos).eq(getattr(e.tok,tr).mask_token_id)
                n=int(mask.sum());mi=None
                if n:
                    q=prob[mask].mean(0);mi=float(-(q*q.clamp_min(1e-30).log2()).sum()-h[mask].mean())
                values,idx=logits.topk(min(self.k,len(valid)),-1)
                row['tracks'][tr]={'entropy_bits':h.cpu().tolist(),'masked':mask.cpu().tolist(),'positions':pos.cpu().tolist(),'mutual_information_bits':mi,'topk_ids':valid[idx].cpu().tolist(),'retained_mass':(values.logsumexp(-1)-logits.logsumexp(-1)).exp().cpu().tolist()}
            self.rows.append(row);return result
        self.model.forward=forward;return self
    def __exit__(self,*a):
        import gzip,json
        self.model.forward=self.original
        with gzip.open(self.out,'wt') as f:json.dump(self.rows,f,allow_nan=False)

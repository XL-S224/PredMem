"""Discrete schedules over actual read opportunities (first forward has no cache)."""
import math
KINDS=('static','static_half','linear_decay','cosine_decay','early_half','late_half')
def factor(kind,step,total):
 assert kind in KINDS and total>=1 and 0<=step<total
 if kind=='static':return 1.
 assert total>=3, 'nonconstant schedules require at least three forward steps'
 if kind=='static_half':return .5
 if step==0:return 0.
 u=(step-1)/(total-2)
 if kind=='linear_decay':return 1-u
 if kind=='cosine_decay':return .5*(1+math.cos(math.pi*u))
 # Exact half summed lambda, including odd numbers of read opportunities.
 middle=(total-2)/2
 early=1. if step-1<middle else .5 if step-1==middle else 0.
 return early if kind=='early_half' else 1-early

def witness():
 for total in [3,4,8,100,300,500]:
  curves={k:[factor(k,t,total) for t in range(total)] for k in KINDS}
  target=.5*(total-1)
  for k in KINDS:
   assert all(0<=x<=1 for x in curves[k])
   if k!='static':assert abs(sum(curves[k][1:])-target)<1e-10,(k,total)
  for k in ['linear_decay','cosine_decay']:
   assert curves[k][1]==1 and curves[k][-1]==0
   assert all(a>=b for a,b in zip(curves[k][1:],curves[k][2:]))
 print('DISCRETE_SCHEDULE_CPU_PASS')
if __name__=='__main__':witness()

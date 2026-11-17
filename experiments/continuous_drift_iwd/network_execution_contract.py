"""Compare original and CUDA-graphed network updates, including AMP/Adam and eval transitions.
Run on CUDA; output records numerical checks, not drift-policy acceptance.
"""
from pathlib import Path
import sys,copy,torch,json,time,gc
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from utils.mirror_policy import SharedMirrorActor,SharedMirrorCritic
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--amp-dtype',choices=['float16','bfloat16','float32'],default='bfloat16');cli=ap.parse_args()
amp=cli.amp_dtype!='float32';dtype=getattr(torch,cli.amp_dtype)
from utils.drift_network import beta_supervision
torch.set_num_threads(2);torch.manual_seed(17);torch.cuda.set_per_process_memory_fraction(.28);results=[]
for kind in ['actor','critic']:
 p=(SharedMirrorActor(7,True) if kind=='actor' else SharedMirrorCritic(preview_count=7)).cuda();q=copy.deepcopy(p);q.execution_backend='cuda_graph'
 if kind=='actor':p.gru.flatten_parameters();q.gru.flatten_parameters()
 op=torch.optim.Adam(p.parameters(),lr=3e-4);oq=torch.optim.Adam(q.parameters(),lr=3e-4)
 errors=[];times=[[],[]];scalers=[torch.amp.GradScaler('cuda'),torch.amp.GradScaler('cuda')]
 for i in range(10):
  mean=torch.randn(43 if kind=='actor' else 72,device='cuda');var=torch.rand_like(mean)+.1
  if kind=='actor':
   x=torch.randn(10,1024,43,device='cuda');h=torch.randn(2,1,1024,128,device='cuda');done=torch.rand(10,1024,device='cuda')<.2;args=(x,mean,var,h,done)
   old=torch.randn(10,1024,3,device='cuda');adv=torch.randn(10,1024,device='cuda');oldlp=torch.randn_like(adv);labels=torch.randn(10240,2,device='cuda');labels[:,1]=labels[:,1].abs()*10
  else:x=torch.randn(10240,72,device='cuda');args=(x,mean,var);target=torch.randn(10240,1,device='cuda')
  grads=[];outs=[]
  for j,(net,opt) in enumerate([(p,op),(q,oq)]):
   opt.param_groups[0]['lr']=3e-4/(1+i*.1);opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter();
   with torch.autocast('cuda',dtype=dtype,enabled=amp):z=net(*args)
   if kind=='actor':
    mu,ls,beta,_=z;dist=torch.distributions.Normal(mu,ls.exp());lp=dist.log_prob(old).sum(-1);ratio=(lp-oldlp).exp().clamp(max=100);loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()+.1*beta_supervision(beta.reshape(-1,1),labels);out=mu
   else:loss=(z-target).square().mean();out=z
   scalers[j].scale(loss).backward();scalers[j].unscale_(opt);grads.append([v.grad.clone() if v.grad is not None else None for v in net.parameters()]);outs.append(out.detach().clone());torch.nn.utils.clip_grad_norm_(net.parameters(),1.);scalers[j].step(opt);scalers[j].update();torch.cuda.synchronize();times[j].append(time.perf_counter()-t)
  ge=max(float(torch.where((a==b)|(torch.isnan(a)&torch.isnan(b)),torch.zeros_like(a),(a-b).abs()).max()) for a,b in zip(*grads) if a is not None);pe=max(float((a-b).abs().max()) for a,b in zip(p.parameters(),q.parameters()));ye=float((outs[0]-outs[1]).abs().max());errors.append(dict(output=ye,gradient=ge,parameter=pe));print(kind,i,errors[-1],flush=True)
  assert pe==0 and ye==0,errors[-1]
  for a,b in zip(*grads):
   if a is not None:torch.testing.assert_close(a,b,atol=0,rtol=0,equal_nan=True)
  assert scalers[0].get_scale()==scalers[1].get_scale()
  for u,v in zip(p.parameters(),q.parameters()):
   for k in op.state[u]:assert torch.equal(op.state[u][k],oq.state[v][k]),('optimizer',kind,i,k)
  assert set(p.state_dict())==set(q.state_dict())
  if kind=='actor':
   p.eval();q.eval()
   with torch.no_grad():
    ref=p(x[:1],mean,var,h,done[:1]);out=q(x[:1],mean,var,h,done[:1]);torch.testing.assert_close(ref[0],out[0],atol=0,rtol=0)
   p.train();q.train()
 row=dict(kind=kind,steps=10,bitwise_equal=True,errors=errors,reference_seconds=times[0],graph_seconds=times[1]);results.append(row)
 del p,q,op,oq,grads,outs,args;gc.collect();torch.cuda.empty_cache()
(Path(cli.output)).write_text(json.dumps(dict(passed=True,results=results),indent=2));print('all exact optimizer checks passed',flush=True)

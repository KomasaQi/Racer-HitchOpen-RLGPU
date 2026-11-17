from pathlib import Path
import sys,copy,time,json,torch,math
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'xcar-simulation')]
from fast_plant import FastRacerPlant,preset_parameters,PARAM_INDEX
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=600);ap.add_argument('--device',default='cuda');ap.add_argument('--backend',choices=['cached','cuda_graph'],default='cuda_graph');ap.add_argument('--output',required=True);args=ap.parse_args()
torch.set_grad_enabled(False);torch.set_num_threads(2);torch.manual_seed(80)
p=FastRacerPlant(preset_parameters('fullsize_racecar_ev'),48,device=args.device);ranges={k:[float(p.col(k)[0])*.85,float(p.col(k)[0])*1.15] for k in ['m','Iz','B','C','bx','cx','by','cy','motor_tau','lsd_torque_ref','lsd_exponent','gear_ratio'] if k in PARAM_INDEX and float(p.col(k)[0])>0};p.randomize(ranges=ranges);p.col('D').copy_(p.col('D')*torch.tensor([.5,.75,1.],device=args.device).repeat_interleave(16));s=p.zero_state();s[:,3]=torch.rand(48,device=args.device)*25;s[:,4]=torch.randn(48,device=args.device)*4;p.init_wheels(s);s[:,6:10]*=torch.rand(48,4,device=args.device)*3
q=copy.deepcopy(p);p.configure_implicit_solver('original');q.configure_implicit_solver(args.backend);u=torch.zeros(48,3,device=args.device);u[:,1]=.25
q.step(s.clone(),u,.05,5);p.reset_guess();q.reset_guess();a=s.clone();b=s.clone();errors=torch.zeros(10,device=args.device);force=0.;t=time.monotonic()
for i in range(args.steps):
 u[:,0]=.2*math.sin(i*.05/1.3);u[:,1]=.15+.12*math.sin(i*.05/2.1);u[:,2]=.15 if i%160>130 else 0.
 a,ea=p.step(a,u,.05,5);b,eb=q.step(b,u,.05,5)
 assert torch.isfinite(a).all() and torch.isfinite(b).all()
 errors=torch.maximum(errors,(a-b).abs().amax(0))
 force=max(force,float((ea['fx_wheel']-eb['fx_wheel']).abs().max()))
 if i%50==49:print(i+1,errors.tolist(), 'seconds',time.monotonic()-t,flush=True)
 assert torch.equal(a,b),(i,errors.tolist())
 # Parameter updates happen in place after graph capture, just as domain randomization does.
 if i%20==19:
  p.randomize(ranges=ranges);q.p.copy_(p.p)
 # Exercise reset of hidden motor/load state, retaining randomized parameters.
 if i%120==119:
  mask=torch.arange(48,device=args.device)%3==i//120%3
  a[mask]=s[mask];b[mask]=s[mask];p.reset_guess(mask);q.reset_guess(mask)
r=dict(passed=True,steps=args.steps,simulated_seconds=args.steps*.05,parallel_cases=48,roads=[.5,.75,1.],randomized=True,max_state_errors=errors.tolist(),max_fx_error=force,wall_seconds=time.monotonic()-t)
Path(args.output).write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)

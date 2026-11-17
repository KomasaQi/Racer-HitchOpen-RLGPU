from pathlib import Path
import sys,json,torch
r=Path(__file__).resolve().parents[2];sys.path[:0]=[str(r),str(r/'utils'),str(r/'xcar-simulation')]
from utils.mirror_policy import SharedMirrorActor,SharedMirrorCritic
from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
for n in [3,5,7,9]:
 for feedback in [False,True]:
  a=SharedMirrorActor(n,feedback);x=torch.randn(4,3,15+4*n);mu,_,beta,h=a(x,None,None);mm,_,bb,hh=a(a.mirror(x),None,None);torch.testing.assert_close(mm,mu*a.mirror.action_sign);torch.testing.assert_close(bb,-beta)
  mu.square().sum().backward();assert all(p.grad is None for p in a.beta.parameters()),'action gradients leaked into beta head'
  a.zero_grad();a(x,None,None)[2].square().sum().backward();assert a.beta[0].weight.grad.abs().sum()>0
  c=SharedMirrorCritic(preview_count=n);z=torch.randn(3,37+5*n);torch.testing.assert_close(c(z,None,None),c(c.mirror(z),None,None))
 cfg=json.loads(Path(sys.argv[1]).read_text());cfg.update(n=3,device='cpu',preview_time={3:'.5,1,1.5',5:'.25,.5,1,1.5,2',7:'.25,.5,1,1.5,2,2.5,3',9:'.25,.5,.75,1,1.25,1.5,2,2.5,3'}[n],sensor_delay_max=0,smooth_curve_extra=.8,steer_second_weight=.2,speed_reward_uncapped=True)
 cfg.pop('ref_mode_cli',None);cfg.pop('num_parallel',None)
 e=ContinuousDriftIWDEnv(preset_name="fullsize_racecar_ev",ref_mode="streaming_random",**cfg);obs=e.reset();assert obs['obs'].shape==(3,15+4*n);assert obs['states'].shape==(3,37+5*n)
 for j in range(8):
  obs,reward,done,info=e.step(torch.tensor([[.05,.3,0.]]).repeat(3,1));assert torch.isfinite(reward).all()
 assert torch.equal(e.obs()['obs'],e.obs()['obs'])
print('Mirror symmetry, stop-gradient, critic separation, 3/5/7/9-point real environment steps passed')


# Controlled reward probes isolate each intervention from state/path differences.
import copy,math
base=copy.deepcopy(e._now);e.step_count[:]=10;e.kappa_ref[:]=0;e.kappa_preview[:]=0
e._now['failed'][:]=0;e._now['signed_pos_diff'][:]=0;e._now['angle_diff'][:]=0
e._now['v'][:]=2*e._reference_speed();e._prev=copy.deepcopy(e._now)
e.speed_reward_uncapped=False;e.reward();capped=e.rew_stats.copy()
e.speed_reward_uncapped=True;e.reward();uncapped=e.rew_stats.copy()
assert abs(uncapped['performance']-2*capped['performance'])<1e-5
for k in capped:
 if k!='performance':assert capped[k]==uncapped[k],k
# Smooth coefficient at straight / half-gain curvature, same raw command.
e._now['raw_delta'][:]=0;e._prev['raw_delta'][:]=0
e.adaptive_smooth=.2;e.smooth_curve_extra=0;e.reward();s0=e.rew_stats['smooth']
e.smooth_curve_extra=.8;e.reward();s1=e.rew_stats['smooth']
e.kappa_ref[:]=.05;e.kappa_preview[:]=.05;e.reward();sh=e.rew_stats['smooth']
assert abs(s1-5*s0)<1e-5 and abs(sh-3*s0)<1e-5
# A constant steering ramp has no second-difference penalty; reversals do.
e._now['raw_delta_step'][:]=.025;e._prev['raw_delta_step'][:]=.025;e.reward();assert e.rew_stats['steer_second']==0
e._prev['raw_delta_step'][:]=-.025;e.reward();assert abs(e.rew_stats['steer_second']+.1)<1e-5
e.step_count[:]=1;e.reward();assert e.rew_stats['steer_second']==0
print('PASS: uncapped speed changes only performance; bell coefficients; steering second difference; initial-step mask')

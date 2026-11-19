"""Online short-horizon model-based actor-critic for the 20 Hz drift task."""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'xcar-simulation')]
from experiments.continuous_drift_iwd.eval import ContinuousDriftIWDEnv, Policy, physical_action
from experiments.continuous_drift_iwd.world_reward import DriftObjective, RewardEnv
from utils.mirror_policy import SharedMirrorActor
from utils.drift_network import beta_supervision


class World(nn.Module):
    def __init__(self, width, dynamic):
        super().__init__()
        self.dynamic = dynamic
        self.net = nn.Sequential(nn.Linear(width+3,256), nn.ELU(), nn.Linear(256,256),
                                 nn.ELU(), nn.Linear(256,dynamic+14))

    def forward(self, x, a):
        y = self.net(torch.cat((x,a),-1))
        k = self.dynamic
        return torch.cat((x[:,:k]+y[:,:k],x[:,k:]),1), y[:,k:]


def update(opt, module, loss, actor=False):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if actor:
        nn.utils.clip_grad_value_(module.parameters(), .1)
    norm = nn.utils.clip_grad_norm_(module.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    return norm.detach()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--wm-horizon',type=int,required=True)
    p.add_argument('--actor-horizon',type=int,required=True)
    p.add_argument('--seed',type=int,default=20261120)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--probe',action='store_true')
    args=p.parse_args()
    start=time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    cfg=json.loads(args.config.read_text(encoding='utf8'))
    cfg['ref_mode']=cfg.pop('ref_mode_cli')
    cfg.update(train=True,random_seed=args.seed,quiet=True,implicit_solver_backend='plant_cached_graph')
    n, steps, batch=(64,32,32) if args.smoke else (256,48,192)
    hwm, horizon=args.wm_horizon,args.actor_horizon
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    (out/'nn').mkdir()
    shutil.copyfile(args.config,out/'env_config.json')
    stored=json.loads((out/'env_config.json').read_text())
    stored['random_seed']=args.seed
    (out/'env_config.json').write_text(json.dumps(stored))
    writer=SummaryWriter(str(out/'summaries'))
    env=RewardEnv('fullsize_racecar_ev',n,'cuda',**cfg)
    objective=DriftObjective(env)
    obs=env.reset()
    assert env.dt==.05 and obs['obs'].shape[1]==51 and obs['states'].shape[1]==82
    actor=SharedMirrorActor(9,True).cuda()
    world=World(133+env.plant.p.shape[1],133).cuda()
    critic=nn.Sequential(nn.Linear(82,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,1)).cuda()
    modules=[world,actor,critic]
    optim=[torch.optim.Adam(m.parameters(),lr=2e-4,betas=(.9,.95)) for m in modules]
    initial=[torch.cat([p.detach().flatten() for p in m.parameters()]).clone() for m in modules]
    hidden=torch.zeros(2,1,n,128,device='cuda')
    noise=torch.zeros(n,3,device='cuda')
    mean=torch.zeros(world.dynamic+env.plant.p.shape[1],device='cuda')
    scale=torch.ones_like(mean)
    replay=deque(maxlen=12)
    durations=dict(rollout=0.,model=0.,actor_critic=0.)
    validation={}
    gamma=.99**2.5
    lam=.95**2.5
    cycles=0
    def pack(obs):
        return torch.cat((obs['obs'],obs['states'],env.plant.p),1)
    def policy(o,h):
        u,_,beta,h=actor._forward_reference(o[None],mean[:51],scale[:51].square(),h)
        return u[0].tanh(),beta[0],h
    max_cycles=2 if args.smoke or args.probe else math.inf
    max_reward_error=0.
    while cycles<max_cycles and (cycles==0 or time.monotonic()-start<1800):
        if cycles and cycles%8==0:env.request_path_refresh()
        tick=time.monotonic()
        data={k:[] for k in ['x','next','a','r','done','h','physical']}
        with torch.no_grad():
            for _ in range(steps):
                x=pack(obs)
                data['h'].append(hidden[:,0].transpose(0,1).half())
                u,_,hidden=policy(obs['obs'],hidden)
                noise=.75*noise+.25*torch.randn_like(noise)
                a=(u+noise).clamp(-1,1)
                obs,r,done,_=env.step(physical_action(a,env))
                data['physical'].append(env.reward_features.clone())
                check=objective(env.reward_features,env.reward_failed)[0]
                err=(check-r.flatten()).abs().max()
                if cycles<2:
                    max_reward_error=max(max_reward_error,float(err))
                    assert max_reward_error<1e-4,max_reward_error
                done=done.flatten().bool()
                for k,v in zip(['x','next','a','r','done'],[x,pack(obs),a,r.flatten(),done]):
                    data[k].append(v.clone())
                hidden=hidden*(~done)[None,None,:,None]
                noise=noise*(~done)[:,None]
        block={k:torch.stack(v) for k,v in data.items()}
        replay.append(block)
        train_n=n*7//8
        if cycles==0:
            warm=block['x'][:,:train_n].reshape(-1,len(mean))
            mean=warm.mean(0)
            scale=warm.std(0).clamp_min(.1)
            hidden.zero_()
        def sample(length,hold=False):
            z=block if hold else random.choice(replay)
            j=torch.randint(train_n if hold else 0,n if hold else train_n,(batch,),device='cuda')
            t=torch.randint(steps-length+1,(batch,),device='cuda')
            return z,t,j
        def model_loss(hold=False):
            z,t,j=sample(hwm,hold)
            sim=(z['x'][t,j]-mean)/scale
            mask=torch.ones(batch,device='cuda')
            loss=0.
            for k in range(hwm):
                idx=(t+k,j)
                sim,physical=world(sim,z['a'][idx])
                r,fail=objective(physical)
                done=z['done'][idx]
                nxt=(z['next'][idx][:,:133]-mean[:133])/scale[:133]
                err=F.smooth_l1_loss(sim[:,:133],nxt,reduction='none').mean(1)
                cost=F.smooth_l1_loss(physical,z['physical'][idx],reduction='none').mean(1)+F.binary_cross_entropy(fail.clamp(1e-6,1-1e-6),done.float(),reduction='none')
                loss=loss+((cost+err*(~done))*mask).sum()/mask.sum().clamp_min(1)
                mask=mask*(~done)
            return loss/hwm
        torch.cuda.synchronize(); measured=time.monotonic()
        durations["rollout"]+=measured-tick
        world.requires_grad_(True)
        for _ in range(2 if args.smoke else 40):
            wl=model_loss()
            update(optim[0],world,wl)
        torch.cuda.synchronize(); model_end=time.monotonic()
        durations["model"]+=model_end-measured
        world.requires_grad_(False)
        for _ in range(1 if args.smoke else 4):
            z,t,j=sample(1)
            raw=z['x'][t,j]
            x=(raw-mean)/scale
            h=z['h'][t,j].transpose(0,1).unsqueeze(1).float()
            critic.requires_grad_(False)
            alive=torch.ones(batch,device='cuda')
            value=lambda state:critic(state[:,51:133]).flatten()
            ret=0.
            states=[]; rewards=[]; safe=[]
            for k in range(horizon):
                states.append(x[:,51:133].detach())
                a,beta,h=policy(x[:,:51]*scale[:51]+mean[:51],h)
                if k==0:aux=beta_supervision(beta,raw[:,51:53]/env.priv_scale[:2])
                x,physical=world(x,a)
                r,fail=objective(physical)
                ret=ret+gamma**k*alive*r
                survive=1-fail
                alive=alive*survive
                rewards.append(r.detach());safe.append(survive.detach())
            last=value(x)
            aloss=-(ret+gamma**horizon*alive*last).mean()+.1*aux
            ag=update(optim[1],actor,aloss,actor=True)
            states=torch.stack(states)
            with torch.no_grad():
                vals=critic(states).squeeze(-1)
                nextv=last.detach();adv=0.;targets=[]
                for k in reversed(range(horizon)):
                    adv=rewards[k]+gamma*safe[k]*nextv-vals[k]+gamma*lam*safe[k]*adv
                    targets.append(vals[k]+adv);nextv=vals[k]
                y=torch.stack(targets[::-1]).flatten()
            critic.requires_grad_(True)
            flat=states.flatten(0,1)
            for _ in range(5):
                for ix in torch.randperm(len(y),device='cuda').chunk(4):
                    closs=F.mse_loss(critic(flat[ix]).flatten(),y[ix])
                    update(optim[2],critic,closs)
        torch.cuda.synchronize()
        durations["actor_critic"]+=time.monotonic()-model_end
        cycles+=1
        if cycles==1 or cycles%10==0 or args.smoke or args.probe:
            with torch.no_grad():
                z,t,j=sample(32,True)
                sim=(z['x'][t,j]-mean)/scale
                alive=torch.ones(batch,dtype=torch.bool,device='cuda')
                for k in range(32):
                    sim,features=world(sim,z['a'][t+k,j])
                    predicted_r,prob=objective(features)
                    eligible=alive.clone()
                    alive=alive & ~z['done'][t+k,j]
                    if k+1 in (1,3,8,13,32) and eligible.any():
                        target=(z['next'][t+k,j,:51]-mean[:51])/scale[:51]
                        row=dict(samples=int(alive.sum()),reward_mae=float((predicted_r-z['r'][t+k,j])[eligible].abs().mean()),
                                 obs_mae=float((sim[:,:51]-target)[alive].abs().mean()) if alive.any() else None)
                        validation[k+1]=row
                        for key,val in row.items():
                            if val is not None:writer.add_scalar(f'heldout/{k+1}/{key}',val,cycles)

        with torch.no_grad(): held=model_loss(True)
        for k,v in dict(model_fit=wl,heldout_model_loss=held,critic_loss=closs,actor_gradient=ag,real_reward=block['r'].mean(),real_done=block['done'].float().mean(),wall_s=time.monotonic()-start).items():
            writer.add_scalar('wm/'+k,float(v),cycles)
        if cycles%10==0:writer.flush()
    weights={'a2c_network.core.'+k:v.cpu() for k,v in actor.state_dict().items()}
    weights.update({'running_mean_std.running_mean':mean[:51].cpu(),'running_mean_std.running_var':scale[:51].square().cpu()})
    # ep30 indexes the 30-minute budget for the common evaluator.
    path=out/'nn'/('wm_ep_1_rew_export.pth' if args.smoke else 'wm_ep_30_rew_export.pth')
    torch.save(dict(model=weights,action_distribution='tanh_normal',updates=cycles*(1 if args.smoke else 4)),path)
    with torch.no_grad():
        policy2=Policy(path,'cuda',execution_backend='reference')
        u=policy(obs['obs'],None)[0]
        export_error=float((u-policy2.act(obs['obs'])).abs().max())
    changes=[float((torch.cat([p.detach().flatten() for p in m.parameters()])-w).abs().max()) for m,w in zip(modules,initial)]
    assert export_error<1e-5 and all(math.isfinite(d) and d>0 for d in changes)
    torch.save(dict(world=world.state_dict(),critic=critic.state_dict(),mean=mean,scale=scale),out/'world_critic.pt')
    stat=dict(wall_s=time.monotonic()-start,cycles=cycles,real_transitions=cycles*n*steps,actor_updates=cycles*(1 if args.smoke else 4),world_horizon=hwm,actor_horizon=horizon,phase_seconds=durations,heldout=validation,real_reward_parity_max=max_reward_error,max_weight_changes=dict(zip(['world','actor','critic'],changes)),export_error=export_error,seed=args.seed)
    (out/'audit.json').write_text(json.dumps(stat,indent=2))
    writer.close()
    print(stat)
if __name__=='__main__':main()


"""Online short-horizon model-based actor-critic for the 20 Hz drift task."""
import argparse
import os
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
from experiments.continuous_drift_iwd.world_replay import SequenceReplay
from utils.mirror_policy import SharedMirrorActor
from utils.drift_network import beta_supervision


class World(nn.Module):
    def __init__(self, width, dynamic):
        super().__init__()
        self.dynamic = dynamic
        self.net = nn.Sequential(nn.Linear(width+3,1024), nn.ELU(), nn.Linear(1024,1024), nn.ELU(), nn.Linear(1024,1024),
                                 nn.ELU(), nn.Linear(1024,dynamic+14))

    def forward(self, x, a):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = self.net(torch.cat((x,a),-1)).float()
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
    p.add_argument('--envs',type=int,default=8192)
    p.add_argument('--model-batch',type=int,default=2048)
    p.add_argument('--policy-batch',type=int,default=4096)
    p.add_argument('--microbatch',type=int,default=2048)
    p.add_argument('--seconds',type=int,default=1800)
    p.add_argument('--probe-cycles',type=int,default=3)
    args=p.parse_args()
    torch.set_float32_matmul_precision('high')
    start=time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    cfg=json.loads(args.config.read_text(encoding='utf8'))
    cfg['ref_mode']=cfg.pop('ref_mode_cli')
    cfg.update(train=True,random_seed=args.seed,quiet=True,implicit_solver_backend='plant_cached_graph')
    n, batch, policy_batch=(128,64,128) if args.smoke else (args.envs,args.model_batch,args.policy_batch)
    micro=min(args.microbatch,policy_batch)
    assert policy_batch%micro==0
    steps=8
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
    replay=SequenceReplay(1000,n)
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
    max_cycles=2 if args.smoke else args.probe_cycles if args.probe else math.inf
    max_reward_error=0.
    collected=model_tokens=actor_steps=wm_steps=0
    credit=0.
    next_save=300
    def export(path):
        weights={'a2c_network.core.'+k:v.detach().cpu() for k,v in actor.state_dict().items()}
        weights.update({'running_mean_std.running_mean':mean[:51].cpu(),'running_mean_std.running_var':scale[:51].square().cpu()})
        tmp=path.with_suffix('.tmp')
        torch.save(dict(model=weights,action_distribution='tanh_normal',updates=actor_steps),tmp)
        os.replace(tmp,path)

    def fit_model():
        seq=replay.sample(hwm,batch)
        target=(seq['x']-mean)/scale
        sim=target[0]
        mask=torch.ones(batch,device='cuda')
        loss=0.
        for k in range(hwm):
            sim,physical=world(sim,seq['a'][k])
            fail=objective.failure_probability(physical)
            done=seq['done'][k]
            err=F.smooth_l1_loss(sim[:,:133],target[k+1,:,:133],reduction='none').mean(1)
            p=fail.clamp(1e-6,1-1e-6)
            cost=F.smooth_l1_loss(physical,seq['physical'][k],reduction='none').mean(1)-torch.where(done,p.log(),torch.log1p(-p))
            loss+=((cost+err*(~done))*mask).sum()/mask.sum().clamp_min(1)
            mask=mask*(~done)
        return loss/hwm

    def micro_update(raw,warm_hidden):
        x=(raw-mean)/scale
        h=warm_hidden.transpose(0,1).unsqueeze(1).float()
        alive=torch.ones(micro,device='cuda');ret=0.
        states=[];rewards=[];safe=[]
        for k in range(horizon):
            states.append(x[:,51:133].detach())
            a,beta,h=policy(x[:,:51]*scale[:51]+mean[:51],h)
            if k==0:aux=beta_supervision(beta,raw[:,51:53]/env.priv_scale[:2])
            x,features=world(x,a)
            r,fail=objective(features)
            ret=ret+gamma**k*alive*r
            alive=alive*(1-fail)
            rewards.append(r.detach());safe.append((1-fail).detach())
        last=critic(x[:,51:133]).flatten()
        loss=-(ret+gamma**horizon*alive*last).mean()+.1*aux
        grads=torch.autograd.grad(loss*micro/policy_batch,tuple(actor.parameters()),allow_unused=True)
        with torch.no_grad():
            xs=torch.stack(states)
            vals=critic(xs).squeeze(-1)
            nextv=last.detach();adv=0.;targets=[]
            for k in reversed(range(horizon)):
                adv=rewards[k]+gamma*safe[k]*nextv-vals[k]+gamma*lam*safe[k]*adv
                targets.append(vals[k]+adv);nextv=vals[k]
        return xs.flatten(0,1),torch.stack(targets[::-1]).flatten(),grads

    def improve_policy():
        seed=replay.sample(8,policy_batch,policy=True)
        # Refresh the current actor's recurrent state on eight real sensor packets.
        with torch.no_grad():
            h=seed['h'].transpose(0,1).unsqueeze(1).float()
            for k in range(8):
                _,_,h=policy(seed['obs'][k],h)
                h=h*(~seed['done'][k])[None,None,:,None]
        critic.requires_grad_(False)
        optim[1].zero_grad(set_to_none=True)
        train_x=[];train_y=[]
        for offset in range(0,policy_batch,micro):
            sl=slice(offset,offset+micro)
            xs,ys,grads=micro_update(seed['x'][sl],h[:,0,sl].transpose(0,1))
            train_x.append(xs);train_y.append(ys)
            for param,g in zip(actor.parameters(),grads):
                if g is not None:
                    if param.grad is None:param.grad=g.detach()
                    else:param.grad.add_(g.detach())
        nn.utils.clip_grad_value_(actor.parameters(),.1)
        ag=nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True)
        optim[1].step()
        critic.requires_grad_(True)
        xs=torch.cat(train_x);y=torch.cat(train_y)
        for _ in range(5):
            for ix in torch.randperm(len(y),device='cuda').chunk(4):
                closs=F.mse_loss(critic(xs[ix]).flatten(),y[ix])
                update(optim[2],critic,closs)
        return ag,closs.detach()

    while cycles<max_cycles and (cycles==0 or time.monotonic()-start<args.seconds):
        if cycles and cycles%48==0:env.request_path_refresh()
        tick=time.monotonic()
        data={k:[] for k in ('x','a','r','done','h','physical')}
        length=48 if not cycles else steps
        with torch.no_grad():
            for _ in range(length):
                x=pack(obs)
                data['h'].append(hidden[:,0].transpose(0,1).half())
                u,_,hidden=policy(obs['obs'],hidden)
                noise=.75*noise+.25*torch.randn_like(noise)
                a=(u+noise).clamp(-1,1)
                obs,r,done,_=env.step(physical_action(a,env))
                if cycles<2:
                    err=(objective(env.reward_features,env.reward_failed)[0]-r.flatten()).abs().max()
                    max_reward_error=max(max_reward_error,float(err))
                    assert max_reward_error<1e-4,max_reward_error
                done=done.flatten().bool()
                for k,v in zip(('x','a','r','done','physical'),(x,a,r.flatten(),done,env.reward_features)):
                    data[k].append(v.clone())
                hidden=hidden*(~done)[None,None,:,None]
                noise=noise*(~done)[:,None]
        block={k:torch.stack(v) for k,v in data.items()}
        replay.insert(block);collected+=length*n
        rollout_reward=block['r'].mean()
        rollout_done=block['done'].float().mean()
        readouts=block['physical']
        # Vehicle response diagnostics on collected transitions, independent of imagined rewards.
        beta20=(readouts[:,:,3].abs()>=math.radians(20))
        powered=(readouts[:,:,6]*env.drift_kappa_target>=.25)&(readouts[:,:,8]>=.15)&beta20
        slide_fraction=powered.float().mean()
        beta_fraction=beta20.float().mean()
        if not cycles:
            warm=block['x'][:,:n*7//8].reshape(-1,len(mean))
            mean=warm.mean(0);scale=warm.std(0).clamp_min(.1);hidden.zero_()
        del data,block,readouts
        measured=time.monotonic()
        durations['rollout']+=measured-tick
        # Reference replay_ratio counts supervised timesteps, not optimizer steps.
        # The first 48-step prefill also earns learning credit.
        credit+=length*n*16
        group_cost=10*batch*hwm
        groups=int(credit//group_cost)
        credit-=groups*group_cost
        if args.smoke:groups=1
        for _ in range(groups):
            t=time.monotonic()
            world.requires_grad_(True)
            for _ in range(10):
                wl=fit_model()
                update(optim[0],world,wl)
                wm_steps+=1
            model_tokens+=group_cost
            world.requires_grad_(False)
            torch.cuda.synchronize();u=time.monotonic()
            durations['model']+=u-t
            ag,closs=improve_policy();actor_steps+=1
            torch.cuda.synchronize();durations['actor_critic']+=time.monotonic()-u
        cycles+=1
        if cycles==1 or cycles%10==0 or args.probe or args.smoke:
            with torch.no_grad():
                seq=replay.sample(32,min(batch,512),hold=True)
                target=(seq['x']-mean)/scale
                sim=target[0]
                alive=torch.ones(sim.size(0),dtype=torch.bool,device='cuda')
                for k in range(32):
                    sim,features=world(sim,seq['a'][k])
                    pred,_=objective(features)
                    valid=alive.clone();alive=alive&~seq['done'][k]
                    if k+1 in (1,3,8,13,32) and valid.any():
                        row=dict(samples=int(alive.sum()),reward_mae=float((pred-seq['r'][k])[valid].abs().mean()),
                                 obs_mae=float((sim[:,:51]-target[k+1,:,:51])[alive].abs().mean()) if alive.any() else None)
                        validation[k+1]=row
                        for key,val in row.items():
                            if val is not None:writer.add_scalar(f'heldout/{k+1}/{key}',val,cycles)
        elapsed=time.monotonic()-start
        for key,v in dict(model_fit=wl,critic_loss=closs,actor_gradient=ag,real_reward=rollout_reward,real_done=rollout_done,real_powered_slide_proxy=slide_fraction,real_beta20=beta_fraction,
                          real_transitions=collected,wm_tokens=model_tokens,wm_updates=wm_steps,actor_updates=actor_steps,
                          replay_transitions=replay.size*n,wall_s=elapsed).items():
            writer.add_scalar('wm/'+key,float(v),cycles)
        if not args.smoke and not args.probe and (cycles==1 or elapsed>=next_save):
            export(out/'latest_policy.pth')
            if elapsed>=next_save:next_save+=300
        writer.flush()
    path=out/'nn'/('wm_ep_1_rew_export.pth' if args.smoke else 'wm_ep_30_rew_export.pth')
    export(path)
    with torch.no_grad():
        policy2=Policy(path,'cuda',execution_backend='reference')
        export_error=float((policy(obs['obs'],None)[0]-policy2.act(obs['obs'])).abs().max())
    changes=[float((torch.cat([p.detach().flatten() for p in m.parameters()])-w).abs().max()) for m,w in zip(modules,initial)]
    assert export_error<1e-5 and all(math.isfinite(d) and d>0 for d in changes)
    torch.save(dict(world=world.state_dict(),critic=critic.state_dict(),mean=mean,scale=scale),out/'world_critic.pt')
    stat=dict(wm_hidden_sizes=[1024,1024,1024],wm_compute='bf16 dense layers; fp32 states, losses and weights',wall_s=time.monotonic()-start,cycles=cycles,real_transitions=collected,actor_updates=actor_steps,model_updates=wm_steps,
              supervised_slots=model_tokens,replay_ratio=model_tokens/collected,envs=n,model_batch=batch,policy_batch=policy_batch,micro=micro,
              replay_capacity=n*1000,replay_filled=n*replay.size,wm_parameters=sum(p.numel() for p in world.parameters()),
              peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,world_horizon=hwm,actor_horizon=horizon,
              phase_seconds=durations,heldout=validation,replay_host_gib=sum(v.numel()*v.element_size() for v in replay.storage.values())/2**30,real_reward_parity_max=max_reward_error,
              max_weight_changes=dict(zip(('world','actor','critic'),changes)),export_error=export_error,seed=args.seed)
    (out/'audit.json').write_text(json.dumps(stat,indent=2))
    writer.close();print(stat)
if __name__=='__main__':main()

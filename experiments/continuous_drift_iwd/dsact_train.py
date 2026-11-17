"""Recurrent asymmetric DSAC-T adapter for the GPU drift task.
Loss follows local DSAC-v2/dsac_v2.py (DSAC-T, arXiv:2310.05858).
Keeps the deployment actor/auxiliary supervision, adds burn-in sequence replay.
This is a task adaptation, not an unmodified SB3/SBX implementation.
"""
from pathlib import Path
import sys,json,argparse,time,copy,math,random,collections
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'utils'),str(ROOT/'xcar-simulation')]
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from utils.mirror_policy import SharedMirrorActor,CriticMirror
from utils.drift_network import beta_supervision
from envs.continuous_drift_iwd import ContinuousDriftIWDEnv

class DistributionalQ(nn.Module):
 def __init__(self,n):
  super().__init__();self.mirror=CriticMirror(n);self.mlp=nn.Sequential(nn.Linear(37+5*n+3,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,64),nn.ELU(),nn.Linear(64,2))
 def forward(self,s,a):
  left=self.mlp(torch.cat((s,a),-1));right=self.mlp(torch.cat((self.mirror(s),a*self.mirror.action_sign),-1));z=(left+right)*.5
  return z[...,0],F.softplus(z[...,1])+.001

def sample_action(mu,logstd):
 logstd=logstd.clamp(-5,2);dist=torch.distributions.Normal(mu,logstd.exp());raw=dist.rsample();a=raw.tanh()
 # Stable tanh change-of-variables correction, in normalized action units.
 lp=(dist.log_prob(raw)-2*(math.log(2)-raw-F.softplus(-2*raw))).sum(-1)
 return a,lp

def distributional_loss(q,std,target,bounded,mean_std):
 sd=std.detach();ratio=(mean_std.square()/(sd.square()+.1)).clamp(.1,10)
 return (ratio*(F.huber_loss(q,target,reduction='none',delta=50)+std*(sd.square()-F.huber_loss(q.detach(),bounded,reduction='none',delta=50))/(sd+.1))).mean()

class Learner:
 def __init__(self,n,device,gamma=.999,batch=256):
  self.actor=SharedMirrorActor(n,True).to(device)
  with torch.no_grad():self.actor.mu.bias.copy_(torch.tensor([0.,0.,-1.],device=device));self.actor.logstd.copy_(torch.log(torch.tensor([.35,.45,.25],device=device)))
  self.target=copy.deepcopy(self.actor).requires_grad_(False);self.actor.gru.flatten_parameters();self.target.gru.flatten_parameters();self.q=nn.ModuleList([DistributionalQ(n),DistributionalQ(n)]).to(device);self.qt=copy.deepcopy(self.q).requires_grad_(False)
  self.rms=RunningMeanStd((15+4*n,)).to(device);self.srms=RunningMeanStd((37+5*n,)).to(device)
  self.ao=torch.optim.Adam(self.actor.parameters(),lr=3e-4);self.qo=torch.optim.Adam(self.q.parameters(),lr=3e-4)
  self.logalpha=nn.Parameter(torch.tensor(math.log(.2),device=device));self.alphao=torch.optim.Adam([self.logalpha],lr=3e-4)
  self.last_actor_loss=0.;self.last_aux=0.;self.gamma=gamma;self.batch=batch;self.iteration=0;self.ema=[None,None];self.tau=.005
 def norms(self):return self.rms.running_mean.float(),self.rms.running_var.float()
 def sequence(self,actor,x,h,done,burn=10):
  mean,var=self.norms()
  with torch.no_grad():_,_,_,h=actor(x[:burn],mean,var,h,done[:burn])
  return actor(x[burn:],mean,var,h.detach(),done[burn:])
 def update(self,chunk,scale):
  ids=torch.randint(chunk['obs'].shape[1],(self.batch,),device=chunk['obs'].device);obs=chunk['obs'][:,ids];state=chunk['states'][10:,ids];hidden=chunk['hidden'][:,:,ids].clone();before=chunk['before'][:,ids]
  action=chunk['actions'][10:,ids];reward=chunk['rewards'][10:,ids];done=chunk['done'][10:,ids]
  sn=self.q[0].mirror.normalize(state,self.srms.running_mean.float(),self.srms.running_var.float())
  with torch.no_grad():
   nm,ns,_,_=self.sequence(self.target,obs,hidden,before);na,nlp=sample_action(nm[1:],ns)
   tq=[q(sn[1:],na) for q in self.qt];select=tq[0][0]<tq[1][0];nextq=torch.minimum(tq[0][0],tq[1][0]);samples=[m+s*torch.randn_like(m).clamp(-3,3) for m,s in tq];nextsample=torch.where(select,samples[0],samples[1]);alpha=self.logalpha.exp().detach()
   target=reward+(~done)*self.gamma*(nextq-alpha*nlp);sample=reward+(~done)*self.gamma*(nextsample-alpha*nlp)
  current=[q(sn[:-1],action) for q in self.q];loss=0
  for i,(q,sd) in enumerate(current):
   mean=sd.detach().mean();self.ema[i]=mean if self.ema[i] is None else (1-self.tau)*self.ema[i]+self.tau*mean
   bounded=q.detach()+(sample-q.detach()).clamp(-3*self.ema[i],3*self.ema[i]);loss=loss+distributional_loss(q,sd,target,bounded,self.ema[i])
  self.qo.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.q.parameters(),10);self.qo.step();actorloss=torch.zeros_like(loss);aux=torch.zeros_like(loss)
  if self.iteration%2==0:
   self.q.requires_grad_(False);mu,ls,beta,_=self.sequence(self.actor,obs[:-1],hidden,before[:-1]);act,lp=sample_action(mu,ls);val=torch.minimum(*[q(sn[:-1],act)[0] for q in self.q]);actorloss=(alpha*lp-val).mean();labels=(state[:-1,:,:2]/scale).reshape(-1,2);aux=beta_supervision(beta.reshape(-1,1),labels)
   self.ao.zero_grad();(actorloss+.1*aux).backward();nn.utils.clip_grad_norm_(self.actor.parameters(),10);self.ao.step();self.q.requires_grad_(True);self.last_actor_loss=float(actorloss.detach());self.last_aux=float(aux.detach())
   self.alphao.zero_grad();aloss=-(self.logalpha*(lp.detach()-3)).mean();aloss.backward();self.alphao.step()
   with torch.no_grad():
    self.actor.logstd.clamp_(-5,2)
    for source,targetnet in [(self.actor,self.target),(self.q,self.qt)]:
     for p,t in zip(source.parameters(),targetnet.parameters()):t.lerp_(p,self.tau)
  self.iteration+=1
  if not torch.isfinite(loss+actorloss+aux):raise FloatingPointError('Non-finite DSAC-T loss')
  return dict(critic_loss=float(loss.detach()),actor_loss=self.last_actor_loss,beta_aux=self.last_aux,alpha=float(self.logalpha.exp().detach()))
 def save(self,path,epoch,frame,seconds):
  sd={'a2c_network.core.'+k:v for k,v in self.actor.state_dict().items()};sd.update({'running_mean_std.'+k:v for k,v in self.rms.state_dict().items()})
  torch.save(dict(model=sd,action_distribution='tanh_normal',algorithm='recurrent_asymmetric_dsac_t',epoch=epoch,frame=frame,training_seconds=seconds,gradient_updates=self.iteration,critic=self.q.state_dict(),critic_target=self.qt.state_dict(),actor_target=self.target.state_dict(),critic_rms=self.srms.state_dict(),actor_optimizer=self.ao.state_dict(),critic_optimizer=self.qo.state_dict(),logalpha=self.logalpha.detach(),alpha_optimizer=self.alphao.state_dict(),mean_std=self.ema),path)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--run-root',required=True);ap.add_argument('--exp-name',required=True);ap.add_argument('--epochs',type=int,default=150);ap.add_argument('--num-parallel',type=int,default=16384);ap.add_argument('--batch-sequences',type=int,default=256);ap.add_argument('--updates-per-chunk',type=int,default=40);ap.add_argument('--seed',type=int,default=20261114);ap.add_argument('--device',default='cuda:0');ap.add_argument('--save-freq',type=int,default=5);a=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(a.seed);random.seed(a.seed)
 cfg=json.loads(Path(a.config).read_text());cfg.update(random_seed=a.seed,sensor_delay_max=0,sensor_dropout_prob=0,train=True);out=Path(a.run_root)/('continuous_drift_iwd_'+a.exp_name);(out/'nn').mkdir(parents=True,exist_ok=True);(out/'env_config.json').write_text(json.dumps(cfg,indent=2));(out/'train_args.json').write_text(json.dumps(vars(a),indent=2));kw=cfg.copy();kw.pop('ref_mode_cli',None);kw.pop('num_parallel',None)
 env=ContinuousDriftIWDEnv('fullsize_racecar_ev',a.num_parallel,a.device,'streaming_random',**kw);learner=Learner(env.preview_count,a.device,cfg.get('gamma',.999),a.batch_sequences);obs=env.reset();h=torch.zeros(2,1,a.num_parallel,128,device=a.device);before=torch.zeros(a.num_parallel,dtype=torch.bool,device=a.device);ring=collections.deque(maxlen=8);writer=SummaryWriter(str(out/'summaries'));frame=0;start=time.monotonic();report=[]
 frames_per_epoch=32768*40;chunk_steps=20;assert frames_per_epoch%(chunk_steps*a.num_parallel)==0
 for epoch in range(1,a.epochs+1):
  rewards=[];stats={};epstart=time.monotonic()
  for _ in range(frames_per_epoch//(chunk_steps*a.num_parallel)):
   chunk={k:[] for k in ['obs','states','before','actions','rewards','done']};chunk['hidden']=h.clone()
   for t in range(chunk_steps):
    with torch.no_grad():
     learner.rms(obs['obs']);learner.srms(obs['states']);mu,ls,_,h=learner.actor(obs['obs'][None],*learner.norms(),h,before[None]);act,_=sample_action(mu[0],ls)
     for k,v in [('obs',obs['obs']),('states',obs['states']),('before',before),('actions',act)]:chunk[k].append(v.clone())
     low=torch.as_tensor(env.action_space.low,device=a.device);high=torch.as_tensor(env.action_space.high,device=a.device);obs,r,done,info=env.step(low+(act+1)*.5*(high-low));done=done.bool()
     if info.get('time_outs',torch.zeros_like(done)).any():raise RuntimeError('Unexpected time truncation: terminal observation bootstrap must be implemented')
     chunk['rewards'].append(r.reshape(-1).clone());chunk['done'].append(done.clone());before=done;h[:,:,done]=0;frame+=a.num_parallel;rewards.append(float(r.mean()));
   for k,v in [('obs',obs['obs']),('states',obs['states']),('before',before)]:chunk[k].append(v.clone())
   for k in ['obs','states','before','actions','rewards','done']:chunk[k]=torch.stack(chunk[k])
   ring.append(chunk)
   if len(ring)>=2:
    for update in range(a.updates_per_chunk):stats=learner.update(random.choice(ring),env.priv_scale[:2])
  wall=time.monotonic()-start;mean_reward=sum(rewards)/len(rewards);row=dict(epoch=epoch,frame=frame,seconds=wall,epoch_seconds=time.monotonic()-epstart,reward_per_step=mean_reward,updates=learner.iteration,**stats);report.append(row)
  for k,v in row.items():
   if k not in ['epoch','frame','seconds']:writer.add_scalar('dsact/'+k,v,frame)
  writer.add_scalar('comparison/reward_per_step_frames',mean_reward,frame);writer.add_scalar('comparison/reward_per_step_epoch',mean_reward,epoch);writer.add_scalar('comparison/reward_per_step_wall_seconds',mean_reward,int(wall))
  for k,v in env.rew_stats.items():writer.add_scalar('reward_terms/'+k,v,frame)
  writer.flush();(out/'progress.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
  if epoch%a.save_freq==0 or epoch==a.epochs:learner.save(out/'nn'/f'last_continuous_drift_iwd_ep_{epoch}_dsact.pth',epoch,frame,wall)
 writer.close()
if __name__=='__main__':main()

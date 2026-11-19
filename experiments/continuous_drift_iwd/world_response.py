"""Same-state finite actuator perturbations; no CarSim solver used here."""
from pathlib import Path
import json,sys,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"xcar-simulation")]
import numpy as np,random,torch
@torch.no_grad()
def validate(ck,policy,model,mean,scale,make_env,act,physical_action):
    n=96;seed=7123
    traces=[];design=None
    for sign in (0,1,-1):
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        env=make_env(ck,n,seed);obs=env.reset();h=None;cmds=[];live=torch.ones(n,device='cuda',dtype=torch.bool)
        for k in range(80):
            if sign==0 or k<60:
                u,h=act(policy.mirror_core,policy,obs['obs'],h)
            else:u=design[k].clone()
            if k==60:
                st=torch.cat((obs['obs'],obs['states'],env.plant.p),1).clone()
                sim=(st-mean)/scale;record={}
                realized=torch.zeros(n,device='cuda');predicted=torch.zeros_like(realized)
            if sign and 60<=k<63:
                ix=torch.arange(n,device='cuda')
                u[ix,ix%2]=(u[ix,ix%2]+sign*.08).clamp(-1,1)
            cmds.append(u.clone())
            if k>=60:
                sim,ret,_=model(sim,u);predicted+=ret
            obs,r,end,_=env.step(physical_action(u,env))
            if h is not None:h=h*(~end.flatten().bool())[None,None,:,None]
            if k>=60:
                realized+=r.flatten();live=live&~end.flatten().bool()
                if k in (64,69,79):record[k-59]=(realized.clone(),predicted.clone(),live.clone())
        if sign==0:design=cmds;initial=st
        else:
            if not torch.equal(initial,st):raise RuntimeError('Unpaired initial state in response test')
            traces.append(record)
    result={}
    for k in (5,10,20):
        a,b=(t[k] for t in traces);dr=a[0]-b[0];dp=a[1]-b[1]
        record={}
        for channel in (0,1):
            mask=a[2]&b[2]&(dr.abs()>.01)&(torch.arange(n,device=dr.device)%2==channel)
            record[['steer','throttle'][channel]]=dict(count=int(mask.sum()),sign_agreement=float((dr[mask]*dp[mask]>0).float().mean()) if mask.any() else None)
        result[k]=record
    return result


def main():
    from experiments.continuous_drift_iwd.train_world import World
    from experiments.continuous_drift_iwd.world_reward import DriftObjective
    from experiments.continuous_drift_iwd.eval import ContinuousDriftIWDEnv,Policy,physical_action
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    run=args.run
    checkpoint,=list((run/'nn').glob('*.pth'))
    policy=Policy(checkpoint,'cuda',execution_backend='reference')
    saved=torch.load(run/'world_critic.pt',map_location='cuda',weights_only=False)
    mean=saved['mean'];scale=saved['scale']
    wm=World(mean.numel(),133).cuda()
    wm.load_state_dict(saved['world'])
    cfg=json.loads((run/'env_config.json').read_text())
    cfg['ref_mode']=cfg.pop('ref_mode_cli')
    objective=None
    def make_env(ck,n,seed):
        nonlocal objective
        env=ContinuousDriftIWDEnv('fullsize_racecar_ev',n,'cuda',**dict(cfg,train=True,random_seed=seed,quiet=True))
        objective=DriftObjective(env)
        return env
    def act(core,pol,obs,h):
        mu,_,_,h=core._forward_reference(obs[None],pol.mean,pol.var,h)
        return mu[0].tanh(),h
    def model(x,a):
        x,features=wm(x,a)
        reward,fail=objective(features)
        return x,reward,fail
    results=validate(checkpoint,policy,model,mean,scale,make_env,act,physical_action)
    (run/'action_response.json').write_text(json.dumps(results,indent=2))
    print(results)
if __name__=='__main__':main()

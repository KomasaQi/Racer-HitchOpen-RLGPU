"""Explicit-checkpoint CarSim-only policy evaluation; no FastPlant evaluation queue."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--config',required=True,help='Matching env_config.json')
    p.add_argument('--reference-path',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--scenes',nargs='+',default=['flat_mu050','flat_mu075','flat_mu100'],choices=['flat_mu050','flat_mu075','flat_mu100'])
    p.add_argument('--open-path',action='store_true',help='Open straight reference, not a closed loop')
    p.add_argument('--seed',type=int,default=20261115)
    p.add_argument('--steps',type=int,default=2400)
    a=p.parse_args()
    if a.steps<1:p.error('steps must be positive')
    checkpoint=Path(a.checkpoint).resolve();config=Path(a.config).resolve();reference=Path(a.reference_path).resolve();out=Path(a.output).resolve()
    for f in [checkpoint,config,reference]:
        if not f.is_file():p.error('Missing input: '+str(f))
    if out.is_relative_to(ROOT):p.error('Use an output directory outside the repository')
    if out.exists() and any(out.iterdir()):p.error('Output must be empty; preserve previous results')
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(config.read_text(encoding='utf-8'));cfg['sensor_delay_max']=0;cfg['sensor_dropout_prob']=0
    effective=out/'env_config.json';effective.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    # Same lock as existing DSH archive runners and LastRun replay publication.
    lock=ROOT.parent/'DSH_Night_2026-11-14/carsim.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('x') as handle:handle.write(str(os.getpid()))
    try:
        import torch
        torch.set_num_threads(1)
        from experiments.carsim_bridge.run_policy_carsim import run
        if a.open_path:
            import numpy as np
            from experiments.continuous_drift_iwd import reference_fixture as rf
            original=rf.install_reference;n=len(np.load(reference))
            def dense(env,path,*,closed=True):
                assert not closed and env.n==1 and env.step_count.item()==0
                env.num_waypoints=n;env.num_paths=1
                env.path_bank=env.path_bank.new_zeros((env.path_bank.shape[0],1,n,4))
                env._path_ds=env._path_ds[:,:1].clone()
                return original(env,path,closed=False)
            rf.install_reference=dense
        for scene in a.scenes:
            run(scene,str(checkpoint),str(effective),steps=a.steps,seed=a.seed,output_dir=str(out/scene),device='cpu',reference_path=str(reference),reference_open=a.open_path)
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()

"""Fixed offline evaluation protocol for the approved overnight matrix. Never trains."""
from pathlib import Path
import argparse,json,sys,os,subprocess,hashlib,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
def checkpoint(folder,epoch):
    files=[p for p in (folder/'nn').glob(f'*ep_{epoch}_*.pth') if '_rew__' not in p.name]
    if len(files)!=1: raise ValueError(f'Expected one checkpoint at {epoch}: {files}')
    return files[0]
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);p.add_argument('--epoch',type=int,required=True);p.add_argument('--suite',choices=['smoke','screen','final','fast'],default='screen');p.add_argument('--smoke-runs',action='store_true');p.add_argument('--fixture');p.add_argument('--scene');p.add_argument('--seed',type=int,default=20261115);p.add_argument('--steps',type=int);a=p.parse_args()
    root=Path(a.root);folder=root/('smoke/runs' if a.smoke_runs else 'runs')/f'continuous_drift_iwd_night_{a.case}';ck=checkpoint(folder,a.epoch)
    out=root/('smoke/evaluations' if a.smoke_runs else 'evaluations')/a.case/f'ep{a.epoch}'/a.suite;out.mkdir(parents=True,exist_ok=True)
    if not a.fixture:
        if a.suite=='smoke':jobs=[('circle','flat_mu075',20261115,40)]
        elif a.suite=='screen':jobs=[('circle','flat_mu075',20261115,1200),('eight','flat_mu100',20261115,1200),('ellipse','flat_mu050',20261115,1200),('straight','flat_mu100',20261115,1200)]
        elif a.suite=='fast':jobs=[('eight',s,20261115,a.steps or (40 if a.smoke_runs else 600)) for s in ['flat_mu050','flat_mu075','flat_mu100']]
        else:jobs=[(f,s,seed,2400 if f in ('eight','ellipse') else 1200) for seed in [20261115,20261116] for f in ['circle','circle_cw','eight','ellipse'] for s in ['flat_mu050','flat_mu075','flat_mu100']]+[('straight','flat_mu100',seed,1200) for seed in [20261115,20261116]]
        rows=[]
        for f,s,seed,steps in jobs:
            dest=out/f'{f}_{s}_seed{seed}';result=dest/'summary.json'
            if result.exists() and (a.suite=='fast' or (dest/'trace.png').exists()):rows.append(dict(fixture=f,scene=s,seed=seed,status='already_complete'));continue
            cmd=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--case',a.case,'--epoch',str(a.epoch),'--suite',a.suite,'--fixture',f,'--scene',s,'--seed',str(seed),'--steps',str(steps)]
            if a.smoke_runs:cmd+=['--smoke-runs']
            dest.mkdir(parents=True,exist_ok=True)
            with (dest/'worker.log').open('w',encoding='utf-8') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                try:code=proc.wait(timeout=1200)
                except subprocess.TimeoutExpired:
                    import psutil
                    try:
                        tree=psutil.Process(proc.pid).children(recursive=True)
                        for child in tree:
                            try:child.kill()
                            except psutil.NoSuchProcess:pass
                    except psutil.NoSuchProcess:pass
                    proc.kill();proc.wait();code=-9
                    log.write('\nWorker exceeded 1200s; its process tree was stopped.\n')
            rows.append(dict(fixture=f,scene=s,seed=seed,status='complete' if code==0 and result.exists() else 'failed',exit_code=code));(out/'manifest.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(rows,indent=2));return int(any(x['status']=='failed' for x in rows))
    import torch,numpy as np
    torch.set_num_threads(1)
    dest=out/f'{a.fixture}_{a.scene}_seed{a.seed}';dest.mkdir(parents=True,exist_ok=True);fixture=root/'inputs/fixtures'/f'{a.fixture}.npy'
    config=json.loads((folder/'env_config.json').read_text());assert config['sensor_delay_max']==0
    if a.suite=='fast':
        mu={'flat_mu050':.5,'flat_mu075':.75,'flat_mu100':1.}[a.scene]
        cmd=[sys.executable,str(ROOT/'experiments/continuous_drift_iwd/eval.py'),str(ck),'--config-from',str(folder),'--device','cpu','--n','8','--num-paths','8','--steps',str(a.steps),'--seed',str(a.seed),'--grip-scale',str(mu),'--spawn-mode','legacy','--equilibrium-reset-prob','0','--legacy-zero-prob','1','--reference-path',str(fixture),'--json-out',str(dest/'summary.json')]
        return subprocess.run(cmd,cwd=ROOT).returncode
    # Native solver and GUI LastRun must never be written concurrently.
    import psutil
    lock=root/'carsim.lock'
    if lock.exists():
        pid=int(lock.read_text())
        if psutil.pid_exists(pid):raise RuntimeError(f'CarSim lock owned by live PID {pid}')
        lock.unlink()
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        from experiments.carsim_bridge.run_policy_carsim import run
        if a.fixture=='straight':
            from experiments.continuous_drift_iwd import reference_fixture as rf
            original=rf.install_reference;n=len(np.load(fixture));geometry={}
            def dense(env,path,*,closed=True):
                assert not closed and env.n==1 and env.step_count.item()==0
                env.num_waypoints=n;env.num_paths=1;env.path_bank=env.path_bank.new_zeros((env.path_bank.shape[0],1,n,4));env._path_ds=env._path_ds[:,:1].clone();geometry.update(origin=env.s[0,:2].cpu().tolist(),heading=float(env.s[0,2]));return original(env,path,closed=False)
            rf.install_reference=dense
        result=run(a.scene,str(ck),str(folder/'env_config.json'),steps=a.steps,seed=a.seed,output_dir=str(dest),device='cpu',reference_path=str(fixture),reference_open=a.fixture=='straight')
        x=np.loadtxt(dest/f'trace_{a.scene}.csv',delimiter=',',skiprows=1)
        if a.fixture=='straight':
            dx=x[:,18:20]-np.array(geometry['origin']);h=geometry['heading'];lat=-np.sin(h)*dx[:,0]+np.cos(h)*dx[:,1];assert np.abs(abs(x[:,5])-abs(lat)).max()<.026;assert (np.cos(h)*dx[:,0]+np.sin(h)*dx[:,1]).max()<4990
        m=(x[:,29]>=4)&np.isfinite(x[:,27]);err=np.rad2deg(np.arctan2(np.sin(x[:,27]-x[:,28]),np.cos(x[:,27]-x[:,28])))
        result.update(fixture=a.fixture,brake_duty=float((x[:,8]>.05).mean()),steer_step_rms_deg=float(np.sqrt(np.mean(np.diff(np.rad2deg(x[:,30]))**2))),beta_mae_deg=float(abs(err[m]).mean()) if m.any() else None,checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest())
        (dest/'summary.json').write_text(json.dumps(result,indent=2))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(3,2,figsize=(12,9));axs[0,0].plot(x[:,18],x[:,19]);axs[0,0].set_title('CG trajectory');axs[0,0].axis('equal')
        for ax,cols,title in [(axs[0,1],[1,23],'speed / reference m/s'),(axs[1,0],[5],'path error m'),(axs[1,1],[27,28],'beta predicted / true rad (pre-action)'),(axs[2,0],[30],'raw steering rad'),(axs[2,1],[7,8],'throttle / brake')]:
            for col in cols:ax.plot(x[:,0],x[:,col],label=str(col),lw=.7)
            ax.set_title(title);ax.grid(alpha=.2);ax.legend()
        fig.tight_layout();fig.savefig(dest/'trace.png',dpi=120);plt.close(fig)
    finally:lock.unlink(missing_ok=True)
    return 0
if __name__=='__main__':raise SystemExit(main())

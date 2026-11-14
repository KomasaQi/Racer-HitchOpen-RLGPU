"""Read-only-code overnight scheduler: fixed matrix, bounded time, failures recorded.
No tuning, no automatic model promotion, no restart of a failed training branch.
"""
from pathlib import Path
import argparse,json,subprocess,sys,os,time,hashlib,collections
ROOT=Path(__file__).resolve().parents[2]
def dump(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2),encoding='utf-8');tmp.replace(p)
def setarg(a,k,v):
    if k in a:a[a.index(k)+1]=str(v)
    else:a.extend([k,str(v)])
def verify(root):
    seal=json.loads((root/'seal.json').read_text())
    for name,digest in seal['files'].items():
        path=Path(name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen file changed: '+name)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=seal['commit']:raise RuntimeError('Wrong repository commit')
    smoke=json.loads((root/'smoke/verified.json').read_text());assert smoke['all_passed']
    return json.loads((root/'plan.json').read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--hours',type=float,default=9.5);p.add_argument('--status',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args();root=Path(a.root).resolve();statefile=root/'overnight_state.json'
    if a.status:
        import psutil
        d=json.loads(statefile.read_text()) if statefile.exists() else {'status':'not_started'}
        d['scheduler_live']=psutil.pid_exists(d.get('pid',-1));print(json.dumps(d,indent=2));return
    plan=verify(root)
    if a.dry_run:print(json.dumps(dict(cases=[x['name'] for x in plan['cases']],hours=a.hours,max_gpu_workers=2,seal_verified=True),indent=2));return
    if statefile.exists():raise RuntimeError('Night already started. Inspect existing PIDs/state; never blindly restart or overwrite.')
    if not 0<a.hours<=10:raise ValueError('Wall budget must be in (0,10] hours')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
    # Avoid unrecorded inherited CarSim diagnostic overrides.
    for k in list(env):
        if k.startswith('CARSIM_'):env.pop(k)
    start=time.time();deadline=start+3600*a.hours;training=[];evaluation=None;pending_eval=collections.deque();seen=set();rows=[];evalrows=[]
    queue=collections.deque(plan['cases']);repeats=collections.deque()
    for c in plan['cases']:
        cc=json.loads(json.dumps(c));cc['name'] += '_seed2';cc['args']=cc['args'][:];setarg(cc['args'],'--seed',20261115);setarg(cc['args'],'--exp-name','night_'+cc['name']);repeats.append(cc)
    state=dict(pid=os.getpid(),status='running',started=start,deadline=deadline,training=rows,evaluations=evalrows,skipped=[]);dump(statefile,state)
    (root/'logs').mkdir(exist_ok=True)
    try:
        while queue or repeats or training or pending_eval or evaluation:
            now=time.time();remaining=deadline-now
            # Train at most two 16384-env branches; CPU-only evaluations reserve GPU headroom.
            if not queue and not training and not pending_eval and not evaluation and repeats:
                # Optional seed replication only after the core matrix is trained.
                c=repeats.popleft();queue.append(c)
            if queue and len(training)<2 and remaining>1800:
                c=queue[0];estimate=plan['estimated_seconds_per_epoch']*(c['target_epoch']-c['start_epoch'])*1.25
                if remaining<estimate+1200:
                    state['skipped'].append(dict(name=c['name'],reason='insufficient wall budget for complete training + evaluation'));queue.popleft()
                else:
                    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
                    state['gpu_free_mib']=free
                    if free>=plan.get('launch_free_mib',4400):
                        queue.popleft();args=c['args'][:];setarg(args,'--run-root',root/'runs');log=(root/'logs'/f"train_{c['name']}.log").open('w',encoding='utf-8');proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'run.py'),*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                        row=dict(name=c['name'],pid=proc.pid,status='running',started=now,target_epoch=c['target_epoch'],fresh=c['fresh']);rows.append(row);training.append((proc,log,row,c));print('train',c['name'],proc.pid,flush=True)
            for proc,log,row,c in list(training):
                folder=root/'runs'/f"continuous_drift_iwd_night_{c['name']}";nn=folder/'nn'
                epochs=[30,60,90] if c['fresh'] else [120,135]
                for ep in epochs:
                    key=(c['name'],ep,'screen')
                    files = [f for f in nn.glob(f'*ep_{ep}_*.pth') if '_rew__' not in f.name]
                    if key not in seen and len(files)==1 and now-files[0].stat().st_mtime>2:
                        seen.add(key);pending_eval.append(key)
                code=proc.poll()
                if code is not None:
                    log.close();row.update(status='complete' if code==0 else 'failed',exit_code=code,finished=now);training.remove((proc,log,row,c))
                    if code==0:
                        for suite in ['final','fast']:
                            key=(c['name'],c['target_epoch'],suite)
                            if key not in seen:seen.add(key);pending_eval.appendleft(key)
                    print('train exit',c['name'],code,flush=True)
            if evaluation:
                proc,log,row=evaluation;code=proc.poll()
                if code is not None:log.close();row.update(status='complete' if code==0 else 'failed',exit_code=code,finished=now);evaluation=None
            if not evaluation and pending_eval and remaining>120:
                name,ep,suite=pending_eval.popleft();log=(root/'logs'/f'eval_{name}_{ep}_{suite}.log').open('w',encoding='utf-8');cmd=[sys.executable,str(ROOT/'experiments/continuous_drift_iwd/night_evaluate.py'),'--root',str(root),'--case',name,'--epoch',str(ep),'--suite',suite];proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);row=dict(name=name,epoch=ep,suite=suite,pid=proc.pid,status='running',started=now);evalrows.append(row);evaluation=(proc,log,row)
            if remaining<=0:break
            if remaining<=1800:
                for c in list(queue)+list(repeats):state['skipped'].append(dict(name=c['name'],reason='wall budget reserved for results'))
                queue.clear();repeats.clear()
            if remaining<=120 and not training and not evaluation:break
            state.update(updated=now,pending_evaluations=list(pending_eval));dump(statefile,state);time.sleep(5)
    finally:
        import psutil
        active=[(p,l,r) for p,l,r,c in training]+([evaluation] if evaluation else [])
        for proc,log,row in active:
            if proc.poll() is None:
                parent=psutil.Process(proc.pid)
                for child in parent.children(recursive=True):
                    try:child.terminate()
                    except psutil.NoSuchProcess:pass
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                row.update(status='wall_budget_stopped',finished=time.time())
            log.close()
        state.update(status='finished',finished=time.time(),pending_evaluations=list(pending_eval));dump(statefile,state)
    print('Night finished. Collect data only; no model promoted.',flush=True)
if __name__=='__main__':main()

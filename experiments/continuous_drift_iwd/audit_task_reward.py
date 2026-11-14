"""Counterfactual reward ordering; does not certify closed-loop drift."""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.continuous_drift_iwd.audit_regressions import make

def main():
    torch.set_num_threads(1)
    e = make(False, beta_ref_mode='geometry_drift')
    e.adaptive_path_scale_m = 1.5
    e.adaptive_speed_penalty = 'none'
    e.adaptive_tracking, e.adaptive_performance, e.adaptive_drift = 1., 1.5, 2.3
    e.adaptive_alive, e.adaptive_smooth, e.adaptive_guidance = .1, .1, 0.
    e.beta_gate_scale = .8
    e.speed_target_mode = 'limit'
    base = {k:v.clone() for k,v in e._now.items()}
    def score(shape, beta=0., slip=.02, throttle=.04, delta=.17,
              pos=0., course=0., vs=1., mu=.75, curvature=1/15, steer_step=0.):
        e.adaptive_shape = shape
        e.plant.tyre_scale.fill_(mu)
        e.kappa_ref.fill_(curvature); e.kappa_preview.fill_(curvature)
        speed=e._reference_speed()
        target=e._reference_beta(e.kappa_ref[:,None],speed).squeeze(1)
        e._now={k:v.clone() for k,v in base.items()}
        for key in ('failed','angle_diff','e_pos','signed_pos_diff'):e._now[key].zero_()
        e._now['signed_pos_diff'].fill_(pos); e._now['angle_diff'].fill_(course)
        e._now['beta']=target*beta; e._now['v']=speed*vs
        e._now['kappa_ref'].fill_(curvature); e._now['kappa_power'][:,2:4]=slip
        e._now['delta'].fill_(delta); e._now['raw_delta'].fill_(delta)
        e._prev={k:v.clone() for k,v in e._now.items()}
        e._prev['delta'].fill_(delta-steer_step)
        e._prev['raw_delta'].fill_(delta-steer_step)
        e.u_cmd[:]=torch.tensor([delta,throttle,0.])
        return float(e.reward().mean())
    cases={'grip':{},'entry':dict(beta=.5,slip=.55,throttle=.4,delta=0,pos=.5,course=.2,vs=.85),
           'drift':dict(beta=1,slip=.55,throttle=.4,delta=-.2)}
    scores={shape:{name:score(shape,**args) for name,args in cases.items()} for shape in ['smooth_v1','smooth_task_v2']}
    v2=scores['smooth_task_v2']
    assert v2['drift'] > v2['entry'] > v2['grip']
    assert abs(v2['drift']-scores['smooth_v1']['drift'])<1e-5
    for kw in [dict(mu=1.),dict(curvature=0.)]:
        for args in cases.values():
            assert abs(score('smooth_v1',**args,**kw)-score('smooth_task_v2',**args,**kw))<1e-5
    partial=[score('smooth_task_v2',beta=b,slip=.55,throttle=.4,delta=-.2) for b in [0,.1,.25,.5,.75,1]]
    assert all(a<b for a,b in zip(partial,partial[1:]))
    assert abs(score('smooth_task_v2',**cases['drift'],vs=1)-score('smooth_task_v2',**cases['drift'],vs=1.4))<1e-5
    # Same instantaneous drift state, different necessary steering rate.
    # Its quality bonus is unchanged in v2; only bounded smoothness costs differ.
    steady=score('smooth_task_v2',**cases['drift'])
    steady_terms=dict(e.rew_stats)
    rapid=score('smooth_task_v2',**cases['drift'],steer_step=.075)
    rapid_terms=dict(e.rew_stats)
    assert abs(steady_terms['optional_drift']-rapid_terms['optional_drift'])<1e-6
    assert 0 < steady-rapid < .15
    legacy_rapid=score('smooth_v1',**cases['drift'],steer_step=.075)
    assert scores['smooth_v1']['drift']-legacy_rapid > steady-rapid
    print(json.dumps({'scores':scores,'partial_beta_progress':partial,'rapid_steer_reward_loss':{'smooth_v1':scores['smooth_v1']['drift']-legacy_rapid,'smooth_task_v2':steady-rapid},'checks':'entry outranks grip; full drift preserved; high-grip and straight unchanged; positive beta progress monotonic; no overspeed deduction'},indent=2))

if __name__=='__main__':main()

"""Physical readouts for the differentiable PPO9 reward in imagined rollouts."""
import torch
from experiments.continuous_drift_iwd.eval import ContinuousDriftIWDEnv


class RewardEnv(ContinuousDriftIWDEnv):
    def reward(self):
        reward=super().reward()
        if self._prev is None:
            return reward
        n=self._now
        target_speed=self._reference_speed().reshape(-1)
        pos=n['signed_pos_diff']
        fade=(1-(pos/(self.beta_gate_scale*self.length_scale)).square()).clamp(0,1)
        target_beta=self._reference_beta(self.kappa_ref[:,None],target_speed).flatten()*fade
        self.reward_features=torch.stack((
            pos/self.adaptive_path_scale_m,n['angle_diff']/self.course_limit,
            n['v']/target_speed.clamp_min(.5),n['beta'],target_beta,n['kappa_ref']/.05,
            n['kappa_power'][:,2:4].mean(1)/self.drift_kappa_target,n['delta']/.25,
            self.u_cmd[:,1],self.u_cmd[:,2],(n['delta']-self._prev['delta'])/.05,
            (n['raw_delta']-self._prev['raw_delta'])/.1,
            self.no_progress_count/self.no_progress_limit,
            (self.step_count.float()/10).clamp_max(2),
        ),1)
        self.reward_failed=n['failed']
        return reward


class DriftObjective:
    def __init__(self,e):
        assert e.adaptive_shape=='smooth_task_v2' and e.adaptive_speed_coupling=='precision'
        assert e.speed_reward_uncapped and e.adaptive_speed_penalty=='none' and not e.progress_speed_credit
        assert not any([e.adaptive_guidance,e.precision_bonus_weight,e.straight_stability_weight,e.steer_second_weight,e.smooth_curve_extra])
        self.e=e

    def __call__(self,f,failed=None):
        e=self.e
        pos,course,ratio,beta,ref,curv,slip,delta,thr,brk,d_delta,d_raw,stalled,age=f.unbind(1)
        def sigmoid_curve(x):
            x=x.clamp(0,1)
            return x.square()*(3-2*x)
        # Smooth analogues of the same three FastPlant termination margins and grace period.
        if failed is None:
            margins=torch.stack((pos.abs()*e.adaptive_path_scale_m/e.offpath_limit,course.abs(),stalled),1)
            fail=(1-(1-torch.sigmoid((margins-1)*18)).prod(1))*torch.sigmoid((age-1.05)*20)
        else:fail=failed
        demand=sigmoid_curve(ref.abs()/.15)
        path=1/(1+pos.square())
        joint=path*(1-course.abs()).clamp(0,1)
        use=ratio.clamp(0,1)
        aligned=beta*ref.sign()
        target=ref.abs().clamp_min(.15)
        counter=-delta*curv.sign()
        thr=thr.clamp(0,1);brk=brk.clamp(0,1)
        gate_slip=sigmoid_curve(slip)
        gate_counter=sigmoid_curve(counter)*((curv.abs()*.05-.02)/.02).clamp(0,1)
        strict=gate_slip*sigmoid_curve(thr/e.drift_throttle_gate)*e._dense_target_gate(beta,ref,.30)*gate_counter
        dense=e._weighted_drift_progress(torch.stack((slip.clamp(0,1),(thr/e.drift_throttle_gate).clamp(0,1),
            e._xcar_target_improvement(aligned,target),counter.clamp(-1,1)),1))
        drift=.5*demand*use*(path*dense+joint*strict)
        quality=1-.75*demand*(1-sigmoid_curve(aligned/target)*gate_slip)
        live=(e.adaptive_alive+joint*quality*(e.adaptive_tracking+e.adaptive_performance*ratio.clamp_min(0))
              +e.adaptive_drift*drift+e.adaptive_smooth/(1+d_delta.square())
              -.05*d_raw.square()/(1+d_raw.square())-.1*brk*(1-ratio).clamp(0,1)).clamp_min(0)
        return live*(1-fail)-e.terminal_cost*fail,fail


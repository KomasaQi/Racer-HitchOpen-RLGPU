# DRIFT: current status after 51 rounds

> **2026-11-13 审计更正：本文件以下“flat_mu 的速度控制器拥有油门”结论已被推翻。**
> `INSTALL_SPEED_CONTROLLER` 只表示模块存在；三个原始 `flat_mu*` 的 `OPT_SC 0` 表示控制器关闭。
> 官方求解器开环实测油门 0 与 0.5 的响应显著不同，转向/制动也有权。旧 `rl_rest`
> 失败来自生成器路径缺少尾反斜杠及误删 Control 参数块，不是必须删除驾驶员控制器。
> 当前权威状态和原始 CSV 见 `../Codex_Drift_Audit_2026-11-13/`；以下正文仅保留为历史记录。

## The blocker, found at round 46-50 and still open

**The CarSim vehicle has its own speed controller and it owns the throttle.**

Evidence, measured:

    throttle 0.0  -> v 4.59   kx_rear -0.009
    throttle 0.3  -> v 5.24   kx_rear -0.007
    throttle 0.6  -> v 5.24   kx_rear -0.007
    throttle 0.9  -> v 5.24   kx_rear -0.007
    throttle 20   -> v 4.05   (earlier probe, identical to 0.6 and 60)
    throttle 60   -> v 4.05

Any positive pedal gives the same speed and the rear longitudinal slip is always NEGATIVE, i.e. the
rear axle is dragged, never driven.  `Run_all.par` line 88 is `INSTALL_SPEED_CONTROLLER`.

**Consequence: every CarSim number reported in this session is withdrawn.** The 30 s survivals, the
13-18 degree sideslips and the 8-12 s drifts were measured with a speed controller holding the car,
not with the policy driving it.  They are not evidence of a powered drift.

## Where the fix stands

| step | state |
|---|---|
| `INSTALL_SPEED_CONTROLLER` added to `CONTROL_BLOCK` in carsim_make_rest.py | DONE |
| regenerated runs written to `D:\carsim_wm_py\rl_rest\flat_mu*` | DONE, 8 control lines removed |
| regenerated simfile contains `PORTS_IMP 1,3` and `PORTS_EXP 1,36` | CONFIRMED present |
| env pointed at `rl_rest` | TRIED, then reverted |
| error when pointed there | `read_configuration` returns `n_export = 0` |
| hand-editing the generated `Run_all.par` | TRIED, CarSim refused, restored from .bak_speedctrl |

So the single open item is why `vs_solver.read_configuration` reports 0 exports for a simfile that
plainly contains `PORTS_EXP 1,36`.  The other two scenarios and the original directories still parse,
so the difference is in the regenerated file rather than in the parser.  Compare the regenerated
simfile with the original field by field, in particular the INPUT path and the SET_MACRO lines.

## What the objective has actually achieved

In the **Python plant only**, drift_ev_v73 epoch 190:

    |beta| 0.11-0.18 rad, |r| 0.60-0.75 rad/s, throttle 0.23-0.29, kx_rear 0.2-1.2,
    steering reversals 0.8-1.2/s, and the rear wheels at 7.8-9.5 m/s surface speed against
    5.8-6.4 at the front.

That is a real rear-drive power slide **in that plant**.  It has never been transferred, because
CarSim has never been given a working throttle.

## The plant also has a defect of its own

The rear wheel had no reflected driveline inertia: with Iw = 0.17 kg m^2 and about 7600 Nm of axle
torque the backward-Euler solve puts the root at 2000 rad/s, and one 50 ms step took the rear wheel
to 203 m/s of surface speed.  `J = Iw + I_motor * gear_ratio^2` is now applied to the driven wheels,
which brought that to 18.8 m/s, but the plant still accelerates about 5x faster than CarSim
(5.25 m/s at 0.95 s against CarSim's 1.20 m/s at 1.05 s), so the two vehicles are not the same vehicle
yet.

## Order of work from here

1. Make `read_configuration` parse the regenerated simfile (compare field by field with the original).
2. Point the env at `rl_rest` and require, as a hard gate:
   throttle 0.3 and 0.6 produce DIFFERENT steady speeds AND kx_rear is POSITIVE.
3. Only then re-run any policy in CarSim.
4. Only then calibrate the Python plant's engine/tyre scale against CarSim open loop.
5. Only then retrain, and judge with the per-step joint criterion (throttle >= 0.30, kx_rear >= 0.25,
   0.20 <= |beta| <= 0.75, no |beta| sweep past 2.0) on duty cycle, not on end-of-episode averages.

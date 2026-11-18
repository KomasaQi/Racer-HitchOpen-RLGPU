"""Windows CUDA wheel solver for the FP32 compact RWD model.

The driver launches onto PyTorch's current stream so the outer control-step
graph captures the native kernel and its PyTorch-managed input buffers.
"""
import ctypes
import torch
from win_nvrtc import check, load_kernel


def solve(plant, lo, hi, locked, context, inertia, old, h, resist, command):
    vx, denom, radius, _, _, bx, cx, ex, peak, fz, blend = context[:11]
    wheels = torch.stack(torch.broadcast_tensors(
        vx, denom, radius, context[12], context[13], bx, cx, ex, peak, fz,
        blend, inertia, old, resist,
    ))
    col = plant.col
    vmin = col("battery_min_voltage_v")
    power = vmin * (col("battery_voc_v") - vmin).clamp_min(0)
    power = power / col("battery_resistance_ohm").clamp_min(1e-6) * col("battery_power_limit_scale")
    nominal = 720. * (936.936 - 720.) / .12
    battery = (power / nominal * col("motor_efficiency")).clamp(.25, 2.)
    motors = torch.stack([
        col("gear_ratio"), col("drive_efficiency"), *context[11], plant._motor_torque_state,
        battery, col("lsd_tau_max"), col("lsd_torque_ref"), col("lsd_exponent").clamp(.05, 1.5),
    ])
    result = torch.empty_like(lo)
    tensors = (wheels, motors, plant.motor_rpm, plant.motor_torque, plant.battery_limit_rpm,
               plant.battery_limit_torque, locked, result)
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("The native wheel solver requires contiguous packed data")
    integers = (old.shape[0], plant.motor_rpm.numel(), plant.battery_limit_rpm.numel(),
                plant.wheel_solver_bracket_iters, plant.wheel_solver_bisection_iters,
                getattr(plant, "wheel_solver_correction_iters", 2))
    scalars = (plant.kappa_peak, plant.slip_force_plateau, h, plant.lsd_ref_rpm)
    args = [ctypes.c_void_p(t.data_ptr()) for t in tensors]
    args.extend(ctypes.c_int(i) for i in integers)
    args.extend(ctypes.c_float(v) for v in scalars)
    params = (ctypes.c_void_p * len(args))(*(ctypes.addressof(arg) for arg in args))
    with torch.cuda.device(old.device):
        driver, function, _, _ = load_kernel(old.device, "wheel_native.cu", "solve")
        stream = ctypes.c_void_p(torch.cuda.current_stream(old.device).cuda_stream)
        blocks = (old.numel() + 127) // 128
        check(driver.cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0, stream, params, None), "wheel solve")
    return result

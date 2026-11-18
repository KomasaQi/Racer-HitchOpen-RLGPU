"""FP32 one-step inference for the validated Windows PPO9 rollout shape."""
import ctypes
import torch
from win_nvrtc import check, load_kernel


def recurrent_step(gru, observations, hidden):
    # The rollout graph records both GEMMs once. The input projection uses
    # TF32; the recurrent projection accumulates ordered full FP32 products.
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        inputs = torch.nn.functional.linear(observations, gru.weight_ih_l0)
        torch.set_float32_matmul_precision("highest")
        recurrent = torch.nn.functional.linear(hidden[0], gru.weight_hh_l0)
    finally:
        torch.set_float32_matmul_precision(precision)
    result = torch.empty_like(hidden)
    tensors = (inputs, recurrent, hidden, gru.bias_ih_l0, gru.bias_hh_l0, result)
    args = [ctypes.c_void_p(t.data_ptr()) for t in tensors]
    args.append(ctypes.c_int(result.numel()))
    params = (ctypes.c_void_p * len(args))(*(ctypes.addressof(arg) for arg in args))
    with torch.cuda.device(result.device):
        driver, function, _, _ = load_kernel(result.device, "gru_step.cu", "gru_step")
        stream = ctypes.c_void_p(torch.cuda.current_stream(result.device).cuda_stream)
        check(driver.cuLaunchKernel(function, (result.numel()+255)//256, 1, 1,
                                    256, 1, 1, 0, stream, params, None), "GRU inference")
    return result, result

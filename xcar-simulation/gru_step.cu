// One FP32 PPO9 inference step, with cuDNN's gate and rounding order.
__device__ float logistic(float v) {
    float exp_value, reciprocal;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(exp_value)
        : "f"(__fmul_rn(v, -1.4426950408889634f)));
    asm("rcp.approx.f32 %0, %1;" : "=f"(reciprocal)
        : "f"(__fadd_rn(1.f, exp_value)));
    return reciprocal;
}

extern "C" __global__ void gru_step(
    const float* input, const float* recurrent, const float* hidden,
    const float* input_bias, const float* recurrent_bias, float* output, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int unit = i % 128;
    const int gate = (i / 128) * 384 + unit;
    float reset = logistic(((input[gate] + input_bias[unit]) + recurrent_bias[unit]) + recurrent[gate]);
    float update = logistic(((input[gate+128] + input_bias[unit+128]) + recurrent_bias[unit+128]) + recurrent[gate+128]);
    float candidate = __fmaf_rn(reset, recurrent[gate+256] + recurrent_bias[unit+256],
                                input[gate+256] + input_bias[unit+256]);
    float activated;
    asm("tanh.approx.f32 %0, %1;" : "=f"(activated) : "f"(candidate));
    output[i] = __fmaf_rn(hidden[i], update, __fmul_rn(1.f - update, activated));
}

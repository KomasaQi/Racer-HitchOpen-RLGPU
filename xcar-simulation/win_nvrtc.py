import ctypes
import os
from functools import lru_cache
from pathlib import Path

import torch


def check(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed with CUDA status {status}")


@lru_cache(maxsize=None)
def load_kernel(device, source_name, entry):
    directory = Path(os.environ["CUDA_PATH"]) / "bin"
    dll_directory = os.add_dll_directory(str(directory))
    nvrtc = ctypes.WinDLL(str(directory / "nvrtc64_120_0.dll"))
    program = ctypes.c_void_p()
    source = Path(__file__).with_name(source_name).read_bytes()
    check(nvrtc.nvrtcCreateProgram(ctypes.byref(program), source, source_name.encode("ascii"), 0, None, None), "create program")
    try:
        major, minor = torch.cuda.get_device_capability(device)
        options = (ctypes.c_char_p * 2)(f"--gpu-architecture=compute_{major}{minor}".encode(), b"--fmad=false")
        status = nvrtc.nvrtcCompileProgram(program, len(options), options)
        size = ctypes.c_size_t()
        if status:
            nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            nvrtc.nvrtcGetProgramLog(program, log)
            raise RuntimeError(log.value.decode(errors="replace"))
        check(nvrtc.nvrtcGetPTXSize(program, ctypes.byref(size)), "get PTX size")
        ptx = ctypes.create_string_buffer(size.value)
        check(nvrtc.nvrtcGetPTX(program, ptx), "read PTX")
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
    driver = ctypes.WinDLL("nvcuda.dll")
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    check(driver.cuModuleLoadData(ctypes.byref(module), ptx), "load module")
    check(driver.cuModuleGetFunction(ctypes.byref(function), module, entry.encode("ascii")), "get kernel")
    return driver, function, module, dll_directory

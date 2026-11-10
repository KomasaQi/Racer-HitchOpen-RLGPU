"""Strip the torch.compile ``_orig_mod.`` prefix from a checkpoint so it can be
restored into an uncompiled model (rl_games compiles the model *after* restore,
so a checkpoint saved from a compiled model cannot be resumed directly).

    python make_resume.py in.pth out.pth
"""
import sys
import torch


def strip(d):
    return {k.replace("_orig_mod.", ""): v for k, v in d.items()}


src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
ck["model"] = strip(ck["model"])
if "assymetric_vf_nets" in ck:
    ck["assymetric_vf_nets"] = strip(ck["assymetric_vf_nets"])
torch.save(ck, dst)
print("wrote", dst, "keys", list(ck.keys()))

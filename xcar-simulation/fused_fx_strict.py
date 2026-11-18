"""Experimental FP32 force fusion with libdevice math and FMA disabled."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

@triton.jit
def _kernel(ptrs, out, s0:tl.constexpr,s1:tl.constexpr,s2:tl.constexpr,s3:tl.constexpr,s4:tl.constexpr,s5:tl.constexpr,s6:tl.constexpr,s7:tl.constexpr,s8:tl.constexpr,s9:tl.constexpr,s10:tl.constexpr,s11:tl.constexpr,s12:tl.constexpr,s13:tl.constexpr,s14:tl.constexpr,s15:tl.constexpr,s16:tl.constexpr,s17:tl.constexpr,s18:tl.constexpr,s19:tl.constexpr,s20:tl.constexpr,s21:tl.constexpr,s22:tl.constexpr,s23:tl.constexpr, N:tl.constexpr, KP:tl.constexpr, AP:tl.constexpr, PL:tl.constexpr, BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 row=i//4;col=i%4
 omega=tl.load(ptrs[0]+row*s0+col*s1,i<N,other=0)
 vx=tl.load(ptrs[1]+row*s2+col*s3,i<N,other=0)
 denom=tl.load(ptrs[2]+row*s4+col*s5,i<N,other=0)
 radius=tl.load(ptrs[3]+row*s6+col*s7,i<N,other=0)
 sy=tl.load(ptrs[4]+row*s8+col*s9,i<N,other=0)
 alpha=tl.load(ptrs[5]+row*s10+col*s11,i<N,other=0)
 bx=tl.load(ptrs[6]+row*s12+col*s13,i<N,other=0)
 cx=tl.load(ptrs[7]+row*s14+col*s15,i<N,other=0)
 ex=tl.load(ptrs[8]+row*s16+col*s17,i<N,other=0)
 peak=tl.load(ptrs[9]+row*s18+col*s19,i<N,other=0)
 fz=tl.load(ptrs[10]+row*s20+col*s21,i<N,other=0)
 blend=tl.load(ptrs[11]+row*s22+col*s23,i<N,other=0)
 sx=tl.div_rn(vx-omega*radius,denom)
 ax=tl.abs(sx)
 zx=ax*(1./KP);zy=tl.abs(alpha)*(1./AP)
 rn=tl.maximum(lib.hypot(zx,zy),1e-9)
 xn=tl.minimum(rn*KP,PL)
 rt=tl.maximum(lib.hypot(ax,tl.abs(sy)),1e-9)
 xt=tl.minimum(rt,PL)
 tn=bx*xn;tt=bx*xt
 sn=lib.sin(cx*lib.atan(tn-ex*(tn-lib.atan(tn))))
 st=lib.sin(cx*lib.atan(tt-ex*(tt-lib.atan(tt))))
 negsign=tl.where(sx>0,-1.,tl.where(sx<0,1.,0.))
 fn=tl.div_rn(negsign*peak*fz*sn*zx,rn)
 ft=tl.div_rn(negsign*peak*fz*st*ax,rt)
 tl.store(out+i,(1.-blend)*fn+blend*ft,i<N)

@torch.library.custom_op("fastplant_fusion::fx_strict",mutates_args=())
def fused_fx_strict(omega:torch.Tensor,context:list[torch.Tensor],kappa_peak:float,alpha_peak:float,plateau:float)->torch.Tensor:
 if not omega.is_cuda:
  from fused_fx import formula
  return formula(omega,context,kappa_peak,alpha_peak,plateau)
 if omega.dtype != torch.float32 or omega.ndim != 2 or omega.shape[1] != 4:
  raise ValueError("strict Fx fusion supports FP32 [cars,4] wheels only")
 values=torch.broadcast_tensors(omega,*context)
 out=torch.empty(omega.shape,device=omega.device,dtype=omega.dtype)
 _kernel[(triton.cdiv(out.numel(),256),)](tuple(values),out,*(s for v in values for s in v.stride()),out.numel(),kappa_peak,alpha_peak,plateau,256,enable_fp_fusion=False)
 return out

@fused_fx_strict.register_fake
def _(omega,context,kappa_peak,alpha_peak,plateau):
 return torch.empty(omega.shape,device=omega.device,dtype=omega.dtype)

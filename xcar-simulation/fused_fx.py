"""Experimental FP32 Fx fusion. Same equations, no reduced solver iterations."""
import torch

def formula(omega,context,kappa_peak,alpha_peak,plateau):
    vx,denom,radius,sy,alpha,bx,cx,ex,peak,fz,blend=context
    sx=(vx-omega*radius)/denom
    ax=sx.abs();zx=ax/kappa_peak;zy=alpha.abs()/alpha_peak
    rn=torch.hypot(zx,zy).clamp(min=1e-9)
    xn=torch.clamp(rn*kappa_peak,max=plateau)
    rt=torch.hypot(ax,sy.abs()).clamp(min=1e-9)
    xt=torch.clamp(rt,max=plateau)
    def shape(x):
        t=bx*x
        return torch.sin(cx*torch.atan(t-ex*(t-torch.atan(t))))
    fn=-torch.sign(sx)*peak*fz*shape(xn)*zx/rn
    ft=-torch.sign(sx)*peak*fz*shape(xt)*ax/rt
    return (1.-blend)*fn+blend*ft

_compiled=torch.compile(formula,backend="inductor",fullgraph=True,options={"triton.cudagraphs":False})

@torch.library.custom_op("fastplant_fusion::fx",mutates_args=())
def fused_fx(omega:torch.Tensor,context:list[torch.Tensor],kappa_peak:float,alpha_peak:float,plateau:float)->torch.Tensor:
    if omega.is_cuda:return _compiled(omega,context,kappa_peak,alpha_peak,plateau)
    return formula(omega,context,kappa_peak,alpha_peak,plateau)

@fused_fx.register_fake
def _(omega,context,kappa_peak,alpha_peak,plateau):
    return torch.empty(omega.shape,device=omega.device,dtype=omega.dtype)

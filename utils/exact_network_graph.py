"""Opt-in CUDA graph replay of original mirror actor operations.
No sampling, optimizer, normalization updates or learning-rate schedule is captured.
Graph cache is not a registered submodule and never changes checkpoint keys.
"""
import torch
from torch import nn

def _make_training_graph(call,args):
    with torch.autocast('cuda',enabled=False,cache_enabled=False):
        return _make_training_graph_impl(call,args)

def _make_training_graph_impl(call,args):
    # Release warmup activations BEFORE graph-private allocations. The stock
    # helper retains its last warmup outputs throughout capture, raising peak VRAM.
    for parameter in call.parameters():parameter.grad=None
    static=tuple(x.detach().clone().requires_grad_(x.requires_grad) for x in args)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out=call(*static);out=out if isinstance(out,tuple) else (out,)
        differentiable=tuple(x for x in out if x.requires_grad)
        inputs=tuple(x for x in static if x.requires_grad)+tuple(call.parameters())
        grads=torch.autograd.grad(differentiable,inputs,tuple(torch.ones_like(x) for x in differentiable),allow_unused=True)
        del out,differentiable,grads
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();torch.cuda.empty_cache()
    return torch.cuda.make_graphed_callables(call,static,num_warmup_iters=0,allow_unused_input=True)

def _cache_for(core):
    signature=tuple((p.data_ptr(),p.dtype,p.device,p.requires_grad) for p in core.parameters())
    cache=getattr(core,'_execution_graph_cache',None)
    if cache is None or getattr(core,'_execution_graph_parameters',None)!=signature:
        cache={};object.__setattr__(core,'_execution_graph_cache',cache)
        object.__setattr__(core,'_execution_graph_parameters',signature)
    return cache

class _ActorCall(nn.Module):
    def __init__(self, core, normalize):
        super().__init__();self.core=core;self.normalize=normalize;self.amp=torch.is_autocast_enabled("cuda");self.amp_dtype=torch.get_autocast_dtype("cuda")
    def forward(self,obs,mean,var,hidden,done):
        torch.clear_autocast_cache()
        with torch.autocast('cuda',enabled=self.amp,dtype=self.amp_dtype,cache_enabled=True):
            mu,ls,beta,h=self.core._forward_reference(obs,mean if self.normalize else None,var if self.normalize else None,hidden,done)
            return mu,ls.clone(),beta,h

def actor_graph_call(core,obs,mean,var,hidden,done):
    if hidden is None:hidden=obs.new_zeros(2,1,obs.shape[1],core.hidden_size)
    if done is None:done=torch.zeros(obs.shape[:2],dtype=torch.bool,device=obs.device)
    if mean is None:mean=obs.new_zeros(obs.shape[-1]);var=obs.new_ones(obs.shape[-1]);normalize=False
    else:normalize=True
    grad=torch.is_grad_enabled()
    key=(tuple(obs.shape),obs.dtype,tuple(hidden.shape),normalize,grad,core.training,hidden.dtype,done.dtype,mean.dtype,var.dtype,torch.is_inference_mode_enabled(),torch.is_autocast_enabled("cuda"),torch.get_autocast_dtype("cuda"),obs.device.index,tuple(x.requires_grad for x in (obs,mean,var,hidden)))
    cache=_cache_for(core)
    args=(obs,mean,var,hidden,done)
    if key not in cache:
        if grad and (getattr(core,'_reference_training_used',False) or any(p.grad is not None for p in core.parameters())):
            return core._forward_reference(obs,mean if normalize else None,var if normalize else None,hidden,done)
        call=_ActorCall(core,normalize);call.training=core.training
        if grad:
            cache[key]=_make_training_graph(call,args)
        else:
            static=tuple(x.detach().clone() for x in args)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):call(*static)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):outputs=call(*static)
            cache[key]=(static,graph,outputs)
    entry=cache[key]
    if grad:return entry(*args)
    static,graph,outputs=entry
    for target,source in zip(static,args):target.copy_(source)
    graph.replay()
    return tuple(x.clone() for x in outputs)

class _CriticCall(nn.Module):
    def __init__(self,core,normalize):
        super().__init__();self.core=core;self.normalize=normalize;self.amp=torch.is_autocast_enabled("cuda");self.amp_dtype=torch.get_autocast_dtype("cuda")
    def forward(self,states,mean,var):
        torch.clear_autocast_cache()
        with torch.autocast('cuda',enabled=self.amp,dtype=self.amp_dtype,cache_enabled=True):
            return self.core._forward_reference(states,mean if self.normalize else None,var if self.normalize else None)

def critic_graph_call(core,states,mean,var):
    if mean is None:mean=states.new_zeros(states.shape[-1]);var=states.new_ones(states.shape[-1]);normalize=False
    else:normalize=True
    grad=torch.is_grad_enabled()
    key=(tuple(states.shape),states.dtype,normalize,grad,core.training,torch.is_inference_mode_enabled(),torch.is_autocast_enabled("cuda"),torch.get_autocast_dtype("cuda"),states.device.index,tuple(x.requires_grad for x in (states,mean,var)))
    cache=_cache_for(core)
    args=(states,mean,var)
    if key not in cache:
        if grad and (getattr(core,'_reference_training_used',False) or any(p.grad is not None for p in core.parameters())):
            return core._forward_reference(states,mean if normalize else None,var if normalize else None)
        call=_CriticCall(core,normalize);call.training=core.training
        if grad:cache[key]=_make_training_graph(call,args)
        else:
            static=tuple(x.detach().clone() for x in args)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):call(*static)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):outputs=call(*static)
            cache[key]=(static,graph,outputs)
    entry=cache[key]
    if grad:return entry(*args)
    static,graph,outputs=entry
    for target,source in zip(static,args):target.copy_(source)
    graph.replay()
    return outputs.clone()

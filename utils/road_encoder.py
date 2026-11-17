"""Current observable preview tokens only; mirror branches share all weights."""
import torch
from torch import nn

class RoadBlock(nn.Module):
    def __init__(self,width=32):
        super().__init__();self.width=width;self.norm=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width);self.out=nn.Linear(width,width)
        self.norm2=nn.LayerNorm(width);self.ff=nn.Sequential(nn.Linear(width,2*width),nn.ELU(),nn.Linear(2*width,width))
    def forward(self,x):
        b,n,_=x.shape;d=self.width//4
        q,k,v=self.qkv(self.norm(x)).reshape(b,n,3,4,d).permute(2,0,3,1,4).unbind(0)
        attn=(q@k.transpose(-1,-2)*(d**-.5)).softmax(-1)
        x=x+self.out((attn@v).transpose(1,2).reshape(b,n,self.width))
        return x+self.ff(self.norm2(x))

class RoadEncoder(nn.Module):
    def __init__(self,points,mode):
        super().__init__();self.group=3 if mode=='attention' else 1; width=32 if mode=='attention' else 64
        if points%self.group:raise ValueError('attention road encoder requires preview count divisible by3')
        self.proj=nn.Sequential(nn.Linear(4*self.group,width),nn.ELU(),nn.Linear(width,width),nn.ELU())
        self.position=nn.Parameter(torch.randn(1,points//self.group,width)*.01)
        self.blocks=nn.Sequential(*([RoadBlock(width),RoadBlock(width)] if mode=='attention' else []))
        self.output=nn.Linear(width,64) if mode=='attention' else nn.Identity()
    def forward(self,points):
        points=points.reshape(points.shape[0],points.shape[1]//self.group,4*self.group)
        encoded=self.blocks(torch.nn.functional.elu(self.proj(points)+self.position)).mean(1)
        return self.output(encoded)

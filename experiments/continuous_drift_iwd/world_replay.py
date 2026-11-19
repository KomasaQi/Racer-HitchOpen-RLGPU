"""Time-major CPU replay for large vector plants. Keep sequences on one car."""
import torch


class SequenceReplay:
    def __init__(self, capacity, envs):
        self.capacity = capacity
        self.envs = envs
        self.storage = {}
        self.end = self.size = 0

    def insert(self, block):
        length = block['x'].shape[0]
        assert 0 < length <= self.capacity
        for k, v in block.items():
            if k not in self.storage:
                self.storage[k] = torch.empty((self.capacity, *v.shape[1:]), dtype=v.dtype, device='cpu')
            first = min(length, self.capacity-self.end)
            self.storage[k][self.end:self.end+first].copy_(v[:first])
            if first < length:
                self.storage[k][:length-first].copy_(v[first:])
        self.end = (self.end+length) % self.capacity
        self.size = min(self.capacity,self.size+length)

    def sample(self, length, batch, hold=False, policy=False):
        # Each target uses the following packet from the same env. Terminal
        # transitions are supervised for hazards, never for a respawn displacement.
        cut = self.envs*7//8
        j = torch.randint(cut if hold else 0,self.envs if hold else cut,(batch,))
        start = torch.randint(self.size-length,(batch,))
        t = (self.end-self.size+start + torch.arange(length+1)[:,None]) % self.capacity
        if policy:
            return dict(h=self.storage['h'][t[0],j].cuda(),
                        obs=self.storage['x'][t[:-1],j,:51].cuda(),
                        done=self.storage['done'][t[:-1],j].cuda(),
                        x=self.storage['x'][t[-1],j].cuda())
        out = {'x':self.storage['x'][t,j].cuda()}
        for k in ('a','done','physical','r'):
            out[k]=self.storage[k][t[:-1],j].cuda()
        return out


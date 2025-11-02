import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)

class TensorNormalization(nn.Module):
    def __init__(self,mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def forward(self,X):
        return normalizex(X,self.mean,self.std)

def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class Layer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
            nn.BatchNorm2d(out_plane)
        )
        self.act = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x

class APLayer(nn.Module):
    def __init__(self,kernel_size):
        super(APLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.AvgPool2d(kernel_size),
        )
        self.act = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)


def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x

class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
        self.bn = bn

    def forward(self, x):
        x_ = self.layer(x)
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y
    
class GN(nn.Module):
    def __init__(self, groups:int, channels:int, 
                 eps:float=1e-5, affine:bool=True):
        super(GN, self).__init__()

        assert channels % groups == 0, 'channels should be evenly divisible by groups'
        self.groups = groups  
        self.channels = channels  
        self.eps = eps 
        self.affine = affine 
        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels))  
            self.shift = nn.Parameter(torch.zeros(channels))  

    def forward(self, x: torch.Tensor):
        x_shape = x.shape 
        batch_size = x_shape[0]
        assert self.channels == x.shape[1]
        x = x.view(batch_size, self.groups, -1)

        mean = x.mean(dim=[-1], keepdim=True)  
        mean_x2 = (x**2).mean(dim=[-1], keepdim=True) 
        var = mean_x2 - mean**2
        x_norm = (x-mean) / torch.sqrt(var+self.eps) 

        if self.affine:
            x_norm = x_norm.view(batch_size, self.channels, -1)  
            x_norm = self.scale.view(1,-1,1)* x_norm + self.shift.view(1,-1,1)  
        x_norm = x_norm.view(x_shape)
        return x_norm


class tdGroupNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdGroupNorm, self).__init__()
        self.gn = GN(int(out_panel/16),out_panel)
        self.seqbn = SeqToANNContainer(self.gn)
        
    def forward(self, x):
        y = self.seqbn(x)
        return y
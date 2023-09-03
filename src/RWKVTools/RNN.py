from typing import Any

from torch import nn
class Model(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._for = self.forward
        self.forward = self.state_forward
        
    def recursiveSetState(self, module, state=None, prefix=''):
        if state is None:
            state = self.getState()
        # recursively go through all the modules and if it has a setState method, call it with the state dict
        for name, child in module.named_children():
            if hasattr(child, 'setState'):
                child.setState(state[prefix + name])
            else:
                self.recursiveSetState(child, state, prefix + name + '.')  # recurse

    def recursiveGetState(self, module, state={}, prefix=''):
        # recursively go through all the modules and if it has a getState method, call it and add the result to the state dict
        for name, child in module.named_children():
            if hasattr(child, 'getState'):
                state[prefix + name] = child.getState()
            else:
                self.recursiveGetState(child, state, prefix + name + '.')

        return state
    
    def setState(self, state):
        self.recursiveSetState(self, state)

    def getState(self):
        return self.recursiveGetState(self)
    
    def state_forward(self, *args, state=None, returnState=False, allLogits=False, **kwargs):
        if state is not None:
            self.recursiveSetState(self, state)
        logits = self._for(*args, **kwargs)

        if not allLogits and not self.training:
            logits = logits[:,-1,:].squeeze()
        if returnState:
            return logits, self.recursiveGetState(self)
        else:
            return logits

    
    
# Pytorch lightning
try:
    import pytorch_lightning as pl
    class LightningModel(pl.LightningModule, Model):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

except:
    pass
import torch


class TimeShift(nn.Module):
    def __init__(self, dims, shiftAmount=1, batch=1 , *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state = torch.zeros(batch,shiftAmount, dims)
        self.shift = shiftAmount
    def forward(self, x):
        tokens = x.shape[-2]
        xapp = torch.cat([self.state.to(x.device, x.dtype), x], dim=-2)
        if self.training:
            pass
        else:
            self.state = xapp[:,-self.shift:,:]
        return xapp[:,:tokens,:]
    
    def setState(self, state):
        self.state = state.clone()

    def getState(self):
        return self.state.clone()

import torch.functional as F

class Short_Mem(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.time_shift1 = TimeShift(args.n_embd, shiftAmount=1, batch=args.micro_bsz)
        self.activation = nn.Sequential(
            nn.Linear(args.n_embd*2, args.n_embd, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        xv = self.activation(torch.cat([self.time_shift1(x),x], dim=-1))
        return  xv

class Long_Mem(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        try :
            dimatt = args.dim_att
        except:
            dimatt = args.n_embd
            args.dim_att = dimatt

        self.head_size = 64
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        self.head_size_divisor = 8

        self.chunk_len = 512
        assert args.ctx_len % self.chunk_len == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)

            # fancy time_decay
            decay_speed = torch.ones(self.n_head)
            for h in range(self.n_head):
                decay_speed[h] = -8 + 7 * (h / (self.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            self.time_faaaa = nn.Parameter(torch.ones(self.n_head) * 0.05)

        self.time_shift = TimeShift(args.n_embd, shiftAmount=1, batch=args.micro_bsz)
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)

        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att)

        self.state = torch.zeros(args.micro_bsz, self.n_head, self.head_size, self.head_size, dtype=torch.float32)

    
    def jit_func(self, x):
        B, TT, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr).view(B, TT, self.n_head, self.head_size).transpose(1, 2)            # BTC -> BHTS
        k = self.key(xk).view(B, TT, self.n_head, self.head_size).transpose(1, 2).transpose(-2, -1) # BTC -> BHTS -> BHST
        v = self.value(xv).view(B, TT, self.n_head, -1).transpose(1, 2)                 # BTC -> BHTS
        g = F.silu(self.gate(xg))

        return r, k, v, g

    
    def jit_func_2(self, r, k, v, g, w, wk, wb, ws, state):
        B, H, TT, S = r.size()
        T = min(self.chunk_len, TT)

        s = state.clone()  # state
        x = torch.zeros(B, H, TT, S, device=r.device, dtype=r.dtype) # output
    
        for i in range(TT // T):
            rr = r[:, :, i*T:i*T+T, :]
            kk = k[:, :, :, i*T:i*T+T]
            vv = v[:, :, i*T:i*T+T, :]

            x[:, :, i*T:i*T+T, :] = ((rr @ kk) * w) @ vv  +  (rr @ s) * wb

            s = ws * s + (kk * wk) @ vv

        
        
        x = x.transpose(1, 2).contiguous().view(B * TT, H*S) # BHTS -> BTHS -> BTC
        x = self.ln_x(x / self.head_size_divisor).view(B, TT, H*S) * g
        return self.output(x), s

    
    def forward(self, x):
        H = self.n_head
        T = min(self.chunk_len, x.shape[-2])

        r, k, v, g = self.jit_func(x)

        w = torch.exp(-torch.exp(self.time_decay.float())).unsqueeze(-1)
        
        u = self.time_faaaa.float().unsqueeze(-1)

################################################################################
########
        ws = w.pow(T).reshape(1, H, 1, 1)

        ind = torch.arange(T-1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, T).pow(ind)

        wk = w.reshape(1, H, 1, T)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, T))
        w = torch.tile(w, [T])
        w = w[:, :-T].reshape(-1, T, 2 * T - 1)
        w = w[:, :, T-1:].reshape(1, H, T, T)
########
################################################################################

        w = w.to(dtype=r.dtype)
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)
        out, state = self.jit_func_2(r, k, v, g, w, wk, wb, ws, self.state.to(dtype=r.dtype, device=r.device))
        if not self.training:
            self.setState(state)
        return out

    def setState(self, state):
        self.state = state.clone()

    def getState(self):
        return self.state.clone()

class Feed_Forward(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = TimeShift(args.n_embd, shiftAmount=1, batch=args.micro_bsz)

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        
        self.key = nn.Linear(args.n_embd,args.n_embd, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.n_embd, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv
    
from typing import Any

import os, math, gc, importlib
from torch import nn
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

method = torch.jit.script_method
method = lambda x: x
module = torch.jit.ScriptModule
module = nn.Module

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

    
class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


# Pytorch lightning
try:
    import pytorch_lightning as pl
    class LightningModel(pl.LightningModule, Model):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

        @property
        def deepspeed_offload(self) -> bool:
            strategy = self.trainer.strategy
            if isinstance(strategy, DeepSpeedStrategy):
                cfg = strategy.config["zero_optimization"]
                return cfg.get("offload_optimizer") or cfg.get("offload_param")
            return False
        
        def configure_optimizers(self):
            args = self.args
            
            lr_decay = set()
            lr_1x = set()
            lr_2x = set()
            lr_3x = set()
            for n, p in self.named_parameters():
                if ("time_mix" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif ("time_decay" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_3x.add(n)
                    else:
                        lr_2x.add(n)
                elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif ("time_first" in n) and (args.layerwise_lr > 0):
                    lr_3x.add(n)
                elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                    lr_decay.add(n)
                else:
                    lr_1x.add(n)

            lr_decay = sorted(list(lr_decay))
            lr_1x = sorted(list(lr_1x))
            lr_2x = sorted(list(lr_2x))
            lr_3x = sorted(list(lr_3x))
            # print('decay', lr_decay)
            # print('1x', lr_1x)
            # print('2x', lr_2x)
            # print('3x', lr_3x)
            param_dict = {n: p for n, p in self.named_parameters()}
            
            if args.layerwise_lr > 0:
                if args.my_pile_stage == 2:
                    optim_groups = [
                        {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                        {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                        {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                    ]
                else:
                    optim_groups = [
                        {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                        {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                        {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                    ]
            else:
                optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

            if args.weight_decay > 0:
                optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
                if self.deepspeed_offload:
                    return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
                return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
            else:
                if self.deepspeed_offload:
                    return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
                return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
            # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

        def training_step(self, batch, batch_idx):
            args = self.args
            if args.my_qa_mask != 1:
                idx, targets = batch
                logits = self(idx)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                # if '0' in os.environ["RWKV_MY_TESTING"]:
                #     print('logits', logits)
                #     torch.set_printoptions(threshold=10000)
                #     print('idx', idx)
                #     exit(0)
            else:
                idx, targets, mask = batch
                mask = mask.view(-1)
                sum_mask = torch.sum(mask).item()
                # if sum_mask == 0:
                #     return torch.tensor([0.0], requires_grad=True)

                logits = self(idx)
                if sum_mask == mask.shape[0]:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                    # print('rank', self.global_rank, 'loss', loss.item())
                else:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
                    # loss_raw = loss
                    loss = torch.sum(loss * mask) / sum_mask

                    # torch.set_printoptions(threshold=10000)
                    # if True: #self.global_rank == 1:
                    #     tmp = ''
                    #     sss = 0
                    #     ccc = 0
                    #     for i in range(mask.shape[0]):
                    #         if mask[i] > 0:
                    #             tmp += str(idx.view(-1)[i].item()) + ','
                    #             sss += loss_raw.view(-1)[i].float().item()
                    #             ccc += 1
                    #     print('rank', self.global_rank, 'loss', loss.item(), 'lavg', sss / ccc)#, 'tmp', tmp, 'input', idx)

            return L2Wrap.apply(loss, logits)

        def training_step_end(self, batch_parts):
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

        def generate_init_weight(self):
            print(
                f"""
    ############################################################################
    #
    # Init model weight (slow for large models)...
    #
    ############################################################################
    """
            )
            m = {}
            for n in self.state_dict():
                p = self.state_dict()[n]
                shape = p.shape

                gain = 1.0
                scale = 1.0
                if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or "model.1." in n or "model.3." in n or "model._orig_mod.1" in n or "model._orig_mod.3" in n:
                    if 'ln_x.weight' in n:
                        try:
                            m = (1+int(n.split('.')[1]))
                            layer_scale = (1+int(n.split('.')[2])) / self.args.n_layer
                            m[n] = (p * 0.0) + (layer_scale ** 0.7)
                        except:
                            layer_scale = (1+int(n.split('.')[3])) / self.args.n_layer
                            m[n] = (p * 0.0) + (layer_scale ** 0.7)
                    else:
                        m[n] = p
                else:
                    if "model.0." in n or "model._orig_mod.0" in n:
                        scale = 1 
                    else:
                        if shape[0] > shape[1]:
                            gain = math.sqrt(shape[0] / shape[1])
                        if 'r' in os.environ["RWKV_MY_TESTING"]:
                            zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']
                        else:
                            zero = [".att.key.", ".att.receptance.", ".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']
                        for kk in zero:
                            if kk in n:
                                scale = 0
                        if n == "head.weight":
                            scale = 0.5
                        if "head_k." in n:
                            scale = 0.1
                        if "head_q." in n:
                            scale = 0

                    print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}")

                    if self.args.accelerator.upper() == "GPU":
                        m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                    else:
                        m[n] = torch.empty((shape[0], shape[1]))

                    if scale == 0:
                        nn.init.zeros_(m[n])
                    elif scale < 0:
                        nn.init.uniform_(m[n], a=scale, b=-scale)
                    else:
                        nn.init.orthogonal_(m[n], gain=gain * scale)

                m[n] = m[n].cpu()
                if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                    m[n] = m[n].half()
                elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                    m[n] = m[n].bfloat16()

                # if n == "emb.weight":
                #     print(m[n])

            gc.collect()
            torch.cuda.empty_cache()
            return m

except:
    pass
import torch


class TimeShift(module):
    def __init__(self, dims, shiftAmount=1, batch=1 , *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state = torch.zeros(batch,shiftAmount, dims)
        self.shift = shiftAmount
    
    @method
    def forward(self, x):
        tokens = x.shape[-2]
        xapp = torch.cat([self.state.to(x.device, x.dtype), x], dim=-2)
        if self.training:
            pass
        else:
            self.state = xapp[:,-self.shift:,:].clone()
        return xapp[:,:tokens,:]
    
    def setState(self, state):
        self.state = state.clone()

    def getState(self):
        return self.state.clone()


# Short Memory
class Short_Mem(module):
    def __init__(self, args, shiftAmount=1):
        super().__init__()
        self.time_shift1 = TimeShift(args.n_embd, shiftAmount=shiftAmount, batch=args.micro_bsz)
        self.activation = nn.Sequential(
            nn.Linear(args.n_embd*2, args.n_embd, bias=False),
            nn.ReLU(),
        )

    @method
    def forward(self, x):
        xv = self.activation(torch.cat([self.time_shift1(x),x], dim=-1))
        return  xv

# RWKV5 attention 
class Long_Mem(module):
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

    @method
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

    @method
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

    @method
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

# FFN
class Feed_Forward(module):
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
        
        self.key = nn.Linear(args.n_embd,args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @method
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv
    
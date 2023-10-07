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
        
    def recursiveSetState(self, module, state=None, prefix='', state2=None, mix=0.0):
        if state is None:
            state = self.getState()
        # recursively go through all the modules and if it has a setState method, call it with the state dict
        for name, child in module.named_children():
            
            if hasattr(child, 'setState'):
                if state2 is None:
                    child.setState(state[prefix + name])
                else:
                    # print((state[prefix + name] - state2[prefix + name]).sum())
                    child.setState(torch.lerp(state[prefix + name], state2[prefix + name], mix))
            
            self.recursiveSetState(child, state, prefix + name + '.', state2=state2, mix=mix)

    def recursiveGetState(self, module, state=None, prefix=''):
        if state is None:
            state = {}
        # recursively go through all the modules and if it has a getState method, call it and add the result to the state dict
        for name, child in module.named_children():
            if hasattr(child, 'getState'):
                state[prefix + name] = child.getState().clone()
            
            self.recursiveGetState(child, state, prefix + name + '.')

        return state
    
    def setState(self, state, state2 = None, mix = 0.0):
        self.recursiveSetState(self, state, state2=state2, mix=mix)

    def getState(self):
        return self.recursiveGetState(self)
    
    def recursiveResetState(self,m):
        for name, child in m.named_children():
            if hasattr(child, 'resetState'):
                
                child.resetState()
            
            self.recursiveResetState(child)

    def resetState(self):
        self.recursiveResetState(self)

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
            if len(batch) == 2:
                idx, targets = batch
                self.resetState()
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
            
            all = self.all_gather(loss)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

            return L2Wrap.apply(loss, logits)


        
except:
    pass
import torch








    
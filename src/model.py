########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, importlib
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn

if importlib.util.find_spec('deepspeed'):
    import deepspeed

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################
import torch.nn as nn
import torch.nn.functional as F


########################################################################################################






    
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.lastlayer = args.n_layer-1

        # if layer_id == 0:
        #     self.ln0 = nn.LayerNorm(args.n_embd)

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        
        from .RWKVTools.modules.LongMem import Long_Mem
        from .RWKVTools.modules.FFN import Feed_Forward
        from .RWKVTools.modules.ShortMem import Short_Mem
        self.att = Long_Mem(args, layer_id)
        self.ffn = Feed_Forward(args, layer_id)
        self.ssm = Short_Mem(args, shiftAmount=1, layer=(layer_id+1))

   
    def forward(self, x):

        l = self.ssm(x)
        l = self.att(self.ln1(l)) + l
        l = self.ffn(self.ln2(l)) + l
        return torch.cat([x,l], dim=-1)



from .RWKVTools.RNN import LightningModel


class RWKV(LightningModel):
    def __init__(self, args):
        super().__init__()
        try:
            self.batches = args.micro_bsz
        except:
            self.batches = 1
            args.micro_bsz = 1

        try:
            args.grad_cp
        except:
            args.grad_cp = 0

        try:
            args.ctx_len
        except:
            args.ctx_len = 1024

        try:
            modelpath = args.load_model

        except:
            modelpath = None
        
        if modelpath:
            file = torch.load(modelpath, map_location="cpu")
            keys = list(file.keys())
            print("keys", keys)
            # remove _orig_mod from keys for compatibility with torch.compile
            newObj = {}
            for key in keys:
                if "_orig_mod." in key:
                    newKey = key.replace("_orig_mod.", "")
                    newObj[newKey] = file[key]
                else:
                    newObj[key] = file[key]
            file = newObj
            keys = list(file.keys())

            # detect model details
            vocab_size, n_embd = file[keys[0]].shape
            args.n_embd = n_embd
            args.vocab_size = vocab_size
            args.dim_ffn = file["blocks.0.ffn.value.weight"].shape[1]
            # model layers are model.2.x.yyy: find highest x
            n_layer = 0
            for key in keys:
                if key.startswith("blocks."):
                    layer = int(key.split(".")[1])
                    if layer > n_layer:
                        n_layer = layer
            args.n_layer = n_layer + 1
        else:
            file = None

        try:
            args.dim_ffn
        except:
            args.dim_ffn = 4 * args.n_embd

        self.args = args



        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        
        self.blocks = nn.Sequential(*[Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd*(1+args.n_layer))
        self.head = nn.Linear(args.n_embd*(1+args.n_layer), args.vocab_size, bias=False)

        
        if file:
            self.load_state_dict(file)

       


    

    def forward(self, idx):
        # if idx is list, make tensor
        if isinstance(idx, list):
            idx = torch.tensor(idx)
        # if idx is int, make tensor
        if isinstance(idx, int):
            idx = torch.tensor([idx])
        
        # if idx is not 3 dim tensor, make it 3 dim
        if len(idx.shape) == 1:
            idx = idx.unsqueeze(0)
            idx = idx.repeat(self.batches, 1)


        args = self.args
        idx = idx.to(self.device)

        
        x = self.emb(idx)
        if args.grad_cp == 1:
            return deepspeed.checkpointing.checkpoint(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.ln_out(x)
        x = self.head(x)
        return x

        
    

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
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n:
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
            else:
                if n == "emb.weight":
                    scale = -1 * self.args.lr_init
                else:
                    if shape[0] > shape[1]:
                        import math
                        gain = math.sqrt(shape[0] / shape[1])
                    zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']
                    for kk in zero:
                        if kk in n:
                            scale = 0
                    if n == "head.weight":
                        scale = 0.5

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
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return m
    
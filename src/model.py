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

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.ln3 = nn.LayerNorm(args.n_embd)
        
        from .RWKVTools.RNN import Short_Mem, Long_Mem, Feed_Forward
        self.att = Short_Mem(args)
        self.longmem = Long_Mem(args, layer_id)
        self.ffn = Feed_Forward(args, layer_id)
   
    def forward(self, x):
        x1 = self.ln1(x*2 + self.att(x))
        x2 = self.ln2(x1+x + self.longmem(x))
        x = self.ln3(x2+x+x1 + self.ffn(x))   
        return x



from .RWKVTools.RNN import LightningModel
class RWKV(LightningModel):
    def __init__(self, args):
        super().__init__()
        try:
            self.batches = args.micro_bsz
        except:
            self.batches = 1
            args.micro_bsz = 1
        self.args = args

        emb = nn.Embedding(args.vocab_size, args.n_embd)
        ln_in = nn.LayerNorm(args.n_embd)
        blocks = nn.Sequential(*[Block(args, i) for i in range(args.n_layer)])
        ln_out = nn.LayerNorm(args.n_embd)
        head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        self.model = nn.Sequential(emb, ln_in, blocks, ln_out, head)
        
        try:
            self.load_state_dict(torch.load(args.model_path, map_location="cuda"))
        except:
            print('From Fresh Model')
            pass


    

    def forward(self, idx):
        # if idx is list, make tensor
        if isinstance(idx, list):
            idx = torch.tensor(idx, dtype=torch.long, device=self.model[0].weight.device)
        # if idx is int, make tensor
        if isinstance(idx, int):
            idx = torch.tensor([idx], dtype=torch.long, device=self.model[0].weight.device)
        
        # if idx is not 3 dim tensor, make it 3 dim
        if len(idx.shape) == 1:
            idx = idx.unsqueeze(0)
            idx = idx.repeat(self.batches, 1)
        args = self.args

        if args.grad_cp == 1:
            return deepspeed.checkpointing.checkpoint(self.model, idx)
        else:
            return self.model(idx)

        
    

    
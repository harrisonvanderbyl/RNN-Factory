from torch import nn

from src.models.model import Experimental

class v5Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.lastlayer = args.n_layer-1

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)    
    
        from .modules.FFN import Feed_Forward
        from .modules.LongMem import Long_Mem
        
        self.ffn = Feed_Forward(args, layer_id)
        self.att = Long_Mem(args, layer_id)
        
  
    def forward(self, x):

        if self.layer_id == 0:
            x = self.ln0(x)

        x = self.att(self.ln1(x)) + x
        x = self.ffn(self.ln2(x)) + x
        return x

class RWKV_v5(Experimental):
    def __init__(self, args):
        super().__init__(args, Block=v5Block)
from .StateModule import StateModule
from torch import nn
from .TimeShift import TimeShift
import torch
class MatForward(StateModule):
    def __init__(self, args, layer_id):
        nn.Module.__init__(self)
        self.args = args
        self.layer_id = layer_id
        
        self.complexsize = args.n_embd
        
        self.key = nn.Linear(args.n_embd,self.complexsize*2, bias=False, dtype=torch.bfloat16)
        self.state = torch.complex(torch.ones(args.micro_bsz, 1, self.complexsize), torch.zeros(args.micro_bsz, 1, self.complexsize))

        self.activation = nn.Sequential(
            nn.Linear(self.complexsize, args.n_embd, bias=False, dtype=torch.bfloat16),
            nn.ReLU(),
        )
        # self.receptance = nn.Linear(self.complexsize, args.n_embd, bias=False, dtype=torch.bfloat16)
        # self.time_shift1 = TimeShift(args.n_embd, shiftAmount=1, batch=args.micro_bsz)
    def resetState(self):
        self.state = torch.complex(torch.ones(self.args.micro_bsz, 1, self.complexsize), torch.zeros(self.args.micro_bsz, 1, self.complexsize))
    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).float()
        
       
        complexval = torch.view_as_complex(k.reshape(B, T, self.complexsize,2))
        rm = torch.abs(complexval)
        complexval = complexval / rm
        complexval = torch.cat([self.state.to(complexval.device), complexval], dim=-2)
        
        kv =  complexval.cumprod(dim=-2)
        # if self.layer_id == 1:
        #     print(complexval[0,:,0])
        # print(kv[0,-1][0]-complexval[0,-1][0])
        self.setState(kv[:,-1:,:])
        kv = kv[:,1:,:]
        kv = kv + complexval[:,1:,:] + x*1j + rm
        kv = (kv.real).relu() * kv * (kv.imag).sigmoid()*1j
        

        # kv = kv * rm
       
        return self.activation(kv.real).pow(2) * (kv.imag).sigmoid()
    
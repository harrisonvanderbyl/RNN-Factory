import torch
from torch import nn

from .StateModule import StateModule

from .TimeShift import TimeShift

from torch.nn import functional as F

# RWKV5 attention 
class Long_Mem(StateModule):
    def __init__(self, args, layer_id):
        try :
            dimatt = args.dim_att
        except:
            dimatt = args.n_embd
            args.dim_att = dimatt

        self.head_size = 64
        self.n_head = args.dim_att // self.head_size
        super().__init__(args.micro_bsz, self.n_head, self.head_size, self.head_size)
        self.args = args
        self.layer_id = layer_id

        
        assert args.dim_att % self.n_head == 0
        self.head_size_divisor = 8

        self.chunk_len = 512
        # assert args.ctx_len % self.chunk_len == 0

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
        
            # fancy time_decay
            decay_speed = torch.ones(self.n_head)
            for h in range(self.n_head):
                decay_speed[h] = -6 + 5 * (h / (self.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # V5-R4 changes
            # https://github.com/BlinkDL/RWKV-LM/commit/5aab658f945ba80745d36c2ab411fb43df3a74f9
            tmp = torch.zeros(self.n_head)
            for h in range(self.n_head):
                tmp[h] = ratio_0_to_1 * (1 - (h / (self.n_head - 1)))
            self.time_faaaa = nn.Parameter(tmp)

        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False, dtype=torch.bfloat16)
        self.time_shift = TimeShift(args.n_embd, shiftAmount=1, batch=args.micro_bsz)
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False, dtype=torch.bfloat16)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False, dtype=torch.bfloat16)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False, dtype=torch.bfloat16)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False, dtype=torch.bfloat16)

        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att)


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
    def _forward_wkbs_chunk(self, T, r, k, v):
        H = self.n_head

        w = torch.exp(-torch.exp(self.time_decay.float())).unsqueeze(-1)

        # V5-R2 changes
        u = self.time_faaaa.float().unsqueeze(-1)
        # u = torch.exp(self.time_first.float()).unsqueeze(-1)

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

        w = w.to(dtype=r.dtype)
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)

        return w, wk, wb, ws
    def forward(self, x):
        H = self.n_head
        T = min(self.chunk_len, x.shape[-2])

        r, k, v, g = self.jit_func(x)

    
        w, wk, wb, ws = self._forward_wkbs_chunk(T, r, k, v)

        out, state = self.jit_func_2(r, k, v, g, w, wk, wb, ws, self.state.to(dtype=r.dtype, device=r.device))
        self.setState(state)
        return out



class RWKVv4Att(StateModule):
    def __init__(self, args,  layer_id):
        try:
            dimatt = args.dim_att
        except:
            dimatt = args.n_embd
            args.dim_att = dimatt
        super().__init__(args.micro_bsz,2,args.dim_att)

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for h in range(args.dim_att):
                decay_speed[h] = -5 + 8 * (h /
                                           (args.dim_att - 1))**(0.7 +
                                                            1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1
                                   for i in range(args.dim_att)]) * 0.5
            import math
            self.time_first = nn.Parameter(
                torch.ones(args.dim_att) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(
                torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(
                torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.time_shift = TimeShift(args.n_embd, shiftAmount=1, batch=args.micro_bsz)


    def wkv_op(self, time_decay, time_first, k, v):
            # // const double vv = v[i + token * emb];
            #     // const double wr1 = aa + exp(float(u[i + emb * offset] + w[i + emb * offset] + k[i + token * emb])) * vv;
            #     // const double wr2 = bb + exp(float(u[i + emb * offset] + w[i + emb * offset] + k[i + token * emb]));
            #     // y[i + token * emb] = (wr1) / (wr2+0.001);
            #     // y[i + token * emb] = (1.0 / (1.0 + exp(float(-r[i + token * emb])))) * y[i + token * emb];
            #     // aa = (aa + exp(float(double(k[i + token * emb]))) * vv) * exp(float(w[i + emb * offset]));
            #     // bb = (bb + exp(float(double(k[i + token * emb])))) * exp(float(w[i + emb * offset]));
                
            
            td = time_decay.exp().neg().exp()
            tf = time_first.exp()

            
            ek = k.exp()
            ekk = ek * tf

            B, T, C = k.size()

            stateb = self.getState()[:, 0].to(k.device, torch.double)
            statec = self.getState()[:, 1].to(k.device, torch.double)

            ekkv = ekk*v

            out = torch.zeros(B, T, C, device=k.device, dtype=torch.double)

            if self.genEmbed:
                stateacc = torch.zeros(T, B, 2, C, device=k.device, dtype=torch.double)
                

            for i in range(T):
                # print(stateb.shape)
                a = stateb + ekkv[:,i].double()
                b = statec + ekk[:,i].double()
                out[:,i,:] = a  / (b + 0.001)

                outb = stateb + ek[:,i].double()*v[:,i].double()
                outc = statec + ek[:,i].double()
                
                outb = td * outb
                outc = td * outc

                stateb = outb
                statec = outc

                if self.genEmbed:
                    stateacc[i,:,0] = stateb.clone().detach()
                    stateacc[i,:,1] = statec.clone().detach()

            if self.genEmbed:
                self.setState(stateacc)
            else:
                self.setState(torch.stack([stateb, statec], dim=1))

            return out.to(k.dtype)


    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)

        sr = torch.sigmoid(r)

        # Enforce bf16 type for kv, as this can be mis init
        # when being called directly via inference
        if k.dtype != torch.bfloat16:
            k = k.to(torch.bfloat16)
        if v.dtype != torch.bfloat16:
            v = v.to(torch.bfloat16)

        

        # Enforce bf16 for self.time_first
        # as this can be mis init when being called directly via inference
        # if self.time_first.dtype != torch.bfloat16:
        #     self.time_first = self.time_first.to(torch.bfloat16)

        # Perform the WKV op via cuda code
        y = self.wkv_op(self.time_decay, self.time_first,
                                  k, v)
        return self.output((sr * y).to(x.dtype))


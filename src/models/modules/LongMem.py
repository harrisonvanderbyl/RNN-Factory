import torch
from torch import nn

from .StateModule import StateModule

from .TimeShift import TimeShift

from torch.nn import functional as F
from torch.utils.cpp_extension import load
# RWKV5 attention

wkv5_cuda = None


class Long_Mem(StateModule):
    def __init__(self, args, layer_id):
        global wkv5_cuda
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
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))


        self.gate = args.linear(args.n_embd, args.dim_att, bias=False, dtype=torch.bfloat16)
        self.time_shift = TimeShift(args.n_embd, shiftAmount=1, batch=args.micro_bsz)
        self.receptance = args.linear(args.n_embd, args.dim_att, bias=False, dtype=torch.bfloat16)
        self.key = args.linear(args.n_embd, args.dim_att, bias=False, dtype=torch.bfloat16)
        self.value = args.linear(args.n_embd, args.dim_att, bias=False, dtype=torch.bfloat16)
        self.output = args.linear(args.dim_att, args.n_embd, bias=False, dtype=torch.bfloat16)

        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att)

        
        HEAD_SIZE = args.dim_att // self.n_head
        
          
        class RWKV_5(torch.autograd.Function):
              
            def forward(ctx, B, T, C, H, state, r, k, v, w, u):
                
                with torch.no_grad():
                    assert HEAD_SIZE == C // H
                    ctx.B = B
                    ctx.T = T
                    ctx.C = C
                    ctx.H = H
                    assert w.dtype == torch.float32
                    assert r.is_contiguous()
                    assert k.is_contiguous()
                    assert v.is_contiguous()
                    assert w.is_contiguous()                            
                    assert u.is_contiguous()                            
                    assert state.is_contiguous()

                y = torch.empty((B, T, C), device=w.device, dtype=r.dtype, memory_format=torch.contiguous_format)
                if r.dtype == torch.bfloat16:
                    wkv5_cuda.forward_bf16(B, T, C, H, state, r, k, v, w, u, y)
                elif r.dtype == torch.float16:
                    wkv5_cuda.forward_fp16(B, T, C, H, state, r, k, v, w, u, y)
                elif r.dtype == torch.float32:
                    wkv5_cuda.forward_fp32(B, T, C, H, state, r, k, v, w, u, y)
                return y, state
                    
        self.RWKV_5 = RWKV_5


    def jit_func(self, x,state):
        B, T, C = x.size()
        
        xx, stateout = self.time_shift(x,state.get(f"blocks.{self.layer_id}.att.time_shift",None)) # Mix x with the previous timestep to produce xk, xv, xr
        
        state[f"blocks.{self.layer_id}.att.time_shift"] = stateout
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        return r, k, v, g, state

    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.reshape(B * T, C)
        
        x = self.ln_x(x / self.head_size_divisor).view(B, T, C)
        x = self.output(x * g)
        return x
    
    def torchwise(self, B:int, T:int, C:int, H:int, s, r, k, v, w, u):
        global wkv5_cuda
        if wkv5_cuda is None:
             wkv5_cuda = load(name="wkv5", sources=["./src/models/modules/cuda/cpuonly.cpp"],
                            verbose=True, extra_cflags=["-O3", "-march=native", "-fPIC", "-H"])
                  
        rm = r.transpose(0,1).contiguous()
        km = k.transpose(0,1).contiguous()
        vm = v.transpose(0,1).contiguous()
        out = torch.zeros(T,B, H, C//H).cpu()
        wkv5_cuda.forward_cpu(B,T,C,H, s, rm, km, vm, w, u, out)
                    
        return out.transpose(0,1).reshape(B, T, C), s

    def forward(self, x, state):
        global wkv5_cuda
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, state = self.jit_func(x,state)

        statein = self.getState(state.get(f"blocks.{self.layer_id}.att",None),x)
        
        if (x.device.type == "cuda"):

            if wkv5_cuda is None:
                # check if rocm
                try:
                    wkv5_cuda = load(name="wkv5", sources=["./src/models/modules/cuda/wkv5_op.cpp", f"./src/models/modules/cuda/wkv5_cuda.cu"],
                            verbose=True, extra_cflags=["-O3", "-march=native", "-H", "-fPIC"], extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={C//H}"])
                except:
                    # compile on rocm devices
                    wkv5_cuda = load(name="wkv5", sources=["./src/models/modules/cuda/wkv5_op.cpp", f"./src/models/modules/cuda/wkv5_cuda.cu"],
                            verbose=True, extra_cflags=["-O3", "-march=native", "-H", "-fopenmp", "-fPIC"], extra_cuda_cflags=["-O3", f"-D_N_={C//H}"])
            x, stateout = self.RWKV_5.apply(B, T, C, H, statein, r, k, v, self.time_decay.float().exp().neg().exp().reshape(self.n_head,-1,1), self.time_faaaa.reshape(self.n_head, -1, 1))
        else:
            x, stateout = self.torchwise(B, T, C, H, statein.float(), r.view(B, T, H, -1).float(), k.view(B, T, H, -1).float(), v.view(B, T, H, -1).float(), self.time_decay.double().exp().neg().exp().reshape(self.n_head,-1).float(), self.time_faaaa.reshape(self.n_head, -1).float())
            
        # x, stateout = self.torchwise(B, T, C, H, statein, r.view(B, T, H, -1).float(), k.view(B, T, H, -1).float(), v.view(B, T, H, -1).float(), self.time_decay.double().exp().neg().exp().reshape(self.n_head,-1).float(), self.time_faaaa.reshape(self.n_head, -1).float())
        state[f"blocks.{self.layer_id}.att"] = stateout
        x = x.reshape(B, T, C)
        out = self.jit_func_2(x.to(g.dtype), g)
        return out,state
    
    def getState(self,state, x):
        if state is None:
            return torch.zeros(x.shape[0], self.n_head, self.head_size, self.head_size, device=x.device, dtype=x.dtype)
        
        return state
  
    def resetState(self):
        self.state = None



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

        self.key = args.linear(args.n_embd, args.dim_att, bias=False)
        self.value = args.linear(args.n_embd, args.dim_att, bias=False)
        self.receptance = args.linear(args.n_embd, args.dim_att, bias=False)
        self.output = args.linear(args.dim_att, args.n_embd, bias=False)
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


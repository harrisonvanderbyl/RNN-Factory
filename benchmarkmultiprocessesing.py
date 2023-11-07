########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from src.tokenizer import neox, world
tokenizer = world
import numpy as np
import types, time, gc
import torch
from src.samplers import sample_logits
args = types.SimpleNamespace()


MODEL_NAME = '/home/harrison/Documents/RNN-Factory/src/3B.pth'
MODEL_NAME = "/home/harrison/Documents/RNN-Factory/src/rwkv-raccoon-1b5.pth"
args.load_model = MODEL_NAME
args.micro_bsz = 5


# from src.models.modules.Linear import InferenceLinear, Quantized, Linear


from src.models import RWKV_v4, RWKV_v5, Experimental

model = RWKV_v5(args)





TEMPERATURE = 0.9
top_p = 0.9

model = model.eval()
model = model.requires_grad_(False)
# model = model.float()
model = model.float()
model = model.cpu()



model.resetState()
def init():

    
    return model

question = "What is the simplest yet deepest meaning of life?"


tokeqs = tokenizer.encode("### Question\n"+question+"\n### Answer:\n")


import gradio as gr


def runmodel(tokens, streams):
    model.resetState()
    

    toks = [tokeqs]*streams
    logits = model.forward(toks)
    
    timee = time.clock_gettime(0)
    newtokens = [[]]*streams
    for i in range(tokens):
        toks = [torch.argmax(logits[j],dim=-1).item() for j in range(streams)]
        newtokens = [newtokens[j] + [toks[j]] for j in range(streams)]
        logits = model.forward([[u] for u in toks])
    otime = time.clock_gettime(0)-timee
    # gc
    torch.cuda.empty_cache()
    del logits
    del toks
    del newtokens
    from gc import collect
    collect()
    return otime

  
from tqdm import tqdm
stats = [
    runmodel(100,i*16) for i in tqdm(range(1,8))
]

# display graph
import matplotlib.pyplot as plt
plt.plot(stats)
plt.ylabel('time')
plt.xlabel('streams')
plt.show()

# display table
import pandas as pd
df = pd.DataFrame(stats)
print(df)
df.to_csv('benchmark.csv')


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


# MODEL_NAME = "3B.pth"
# # MODEL_NAME = "/home/harrison/Documents/RNN-Factory/src/rwkv-raccoon-1b5.pth"
# args.load_model = MODEL_NAME
args.micro_bsz = 5
import inquirer
questions = [
  inquirer.List('device',
                message="Which device do you want to use?",
                choices=['cpu', 'cuda'],
            ),
            # choose between bfloat16/float32
            inquirer.List('precision',
                message="Which precision do you want to use?",
                choices=['bfloat16', 'float32'],
            ),
            inquirer.List('size',
                message="How big do you want the model to be?",
                choices=['7B', '3B', '1.5B'],
            ),
]

answers = inquirer.prompt(questions)


# from src.models.modules.Linear import InferenceLinear, Quantized, Linear


from src.models import RWKV_v4, RWKV_v5, Experimental
args.linear = torch.nn.Linear


args.vocab_size = 2**16

if answers['size'] == '3B':
    args.n_layer = 32
    args.n_embd = 2560
    args.n_head = 40

if answers['size'] == '7B':
    args.n_layer = 40
    args.n_embd = 4096
    args.n_head = 64

if answers['size'] == '1.5B':
    args.n_layer = 24
    args.n_embd = 2048
    args.n_head = 32

model = RWKV_v5(args)
# choose between cpu/gpu

device = answers['device']
precision = answers['precision']


TEMPERATURE = 0.9
top_p = 0.9

model = model.eval()
model = model.requires_grad_(False)
# model = model.float()

if precision == 'bfloat16':
    model = model.bfloat16()
else:
    model = model.float()

if device == 'cuda':
    model = model.cuda()



def init():

    
    return model

question = "What is the simplest yet deepest meaning of life?"


tokeqs = tokenizer.encode("### Question\n"+question+"\n### Answer:\n")


import gradio as gr


def runmodel(tokens, streams):
    

    toks = [tokeqs]*streams
    
    logits, state = model.forward(toks, state=None)
    
    timee = time.clock_gettime(0)
    newtokens = [[]]*streams
    import tqdm
    for i in tqdm.tqdm(range(tokens)):
        toks = [torch.argmax(logits[j],dim=-1).item() for j in range(streams)]
        newtokens = [newtokens[j] + [toks[j]] for j in range(streams)]
        logits, _ = model.forward([[u] for u in toks], state=state)
        
    otime = time.clock_gettime(0)-timee
    otime = (tokens*streams)/otime
    tps = tokens/otime
    # gc
    torch.cuda.empty_cache()
    del logits
    del toks
    del newtokens
    from gc import collect
    collect()
    return otime, tps

samples = 10
increase = 64
  
from tqdm import tqdm
stats = [
    runmodel(100,int(1 if i == 0 else i*increase)) for i in tqdm(range(0,samples))
]

fullTokens = sum([i[0] for i in stats])
perStreamTokens = fullTokens/len(stats)

# display graph
import matplotlib.pyplot as plt
plt.plot(stats)
plt.ylabel('Absolute tokens per second')
plt.xlabel('Concurrent Inference Streams')
plt.title('RWKVv5 multi-stream inference')
plt.xticks(range(0,samples),[str(int(1 if i == 0 else i*increase)) for i in range(0,samples)])
plt.ylim(bottom=0)
plt.savefig('benchmark.png')

# display table
import pandas as pd
df = pd.DataFrame(stats)
print(df)
df.to_csv('benchmark.csv')


import copy
import os, gc, torch
import random
import time
import types
#from huggingface_hub import hf_hub_download
from pynvml import *
from torch.nn import functional as F
import numpy as np
import json
# nvmlInit()
# gpu_h = nvmlDeviceGetHandleByIndex(0)
ctx_limit = 8192

torch.set_num_threads(60)

from src.samplers import sample_logits

from src.models import RWKV_v4, RWKV_v5, Experimental
args = types.SimpleNamespace()

# get model file from command line
import sys

if len(sys.argv) < 1:
    print("Please specify port")
    print("Please specify model name")
    print("Pass in optional argument for specific model url")
    print("Example: python3 8080 ./api.py 1B5")
    print("Example: python3 8080 ./api.py 1B5 https://huggingface.co/BlinkDL/rwkv-5-world/resolve/main/RWKV-5-World-1B5-v2-20231025-ctx4096.pth")
    exit()

print(sys.argv)

args.load_model = sys.argv[2] # python3 ./api.py 8080 ./model.pth
PORT = int(sys.argv[1])

# if loadmodel not in 1B5, 7B, 3B throw error
if args.load_model not in ["1B5", "7B", "3B"]:
    print("Invalid model name, options are 1B5, 7B, 3B")
    print("Pass in optional argument for specific model url")
    exit()

# if not exists, download from huggingface
if not os.path.exists(args.load_model+".pth"):
    # if specific url is passed in
    if len(sys.argv) > 3:
        url = sys.argv[3]
    else:
        defaults = {
            "1B5": "https://huggingface.co/BlinkDL/rwkv-5-world/resolve/main/RWKV-5-World-1B5-v2-20231025-ctx4096.pth",
            "7B": "https://huggingface.co/BlinkDL/temp/resolve/main/RWKV-5-World-7B-v2-OnlyForTest_60%25_trained-20231124-ctx4096.pth",
            "3B": "https://huggingface.co/BlinkDL/rwkv-5-world/resolve/main/RWKV-5-World-3B-v2-20231113-ctx4096.pth"
        }
        
        # download and rename
        url = defaults[args.load_model]
    print("Downloading model from", url)
    os.system("wget " + url + " -O " + args.load_model + ".pth")
    print("Downloaded model to", args.load_model)
    
args.load_model = args.load_model + ".pth"

args.withPipeline = True
model = RWKV_v5(args).cuda()
models = [
    model,
]

pipelines = []

from rwkv.utils import PIPELINE, PIPELINE_ARGS
from src.tokenizer import neox, world, racoon

lockedModels = []

for model in models:
    pipelines.append(racoon)
    lockedModels.append(False)

# set thread count with pytorch

import asyncio

async def getModel():
    return models[0], pipelines[0], 0
        
def unlockModel(i):
    lockedModels[i] = False

def sample_logits_typical(logits, temperature=1.0, top_p=0.95, **kwargs):
        probs = F.softmax(logits.float(), dim=-1)
        logits = -torch.log(probs)
        ent = torch.nansum(logits * probs, dim=-1, keepdim=True)
        shifted_logits = torch.abs(logits - ent)
        sorted_ids = torch.argsort(shifted_logits)
        sorted_logits = shifted_logits[sorted_ids]
        sorted_probs = probs[sorted_ids]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = np.sum(cumulative_probs < top_p)
        probs[shifted_logits > sorted_logits[cutoff]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)

def sample_logits(logits, temperature=1.0, top_p=0.85, top_k=0):
    probs = F.softmax(logits.float(), dim=-1)
    top_k = int(top_k)
    if probs.device == torch.device('cpu'):
        probs = probs.numpy()
        sorted_ids = np.argsort(probs)
        sorted_probs = probs[sorted_ids][::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        probs = probs / np.sum(probs)
        out = np.random.choice(a=len(probs), p=probs)
        return int(out)
    else:
        sorted_ids = torch.argsort(probs)
        sorted_probs = probs[sorted_ids]
        sorted_probs = torch.flip(sorted_probs, dims=(0,))
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)
    
thingsToDo = []
mystate = None

def mergestates(states):
    global mystate
    if mystate is None or len(states) > mystate[list(mystate.keys())[0]].shape[0]:
        keys = states[0].keys()
        mystate = {key: torch.cat([state[key].cuda() for state in states], dim=0) for key in keys}
    else:
        for key in states[0].keys():
            for i, state in enumerate(states):
                mystate[key][i] = state[key][0].to(mystate[key].device)

    if len(states) < mystate[list(mystate.keys())[0]].shape[0]:
        keys = states[0].keys()
        mystate = {key: mystate[key][:len(states)] for key in keys}

    return mystate

def splitstates(state):
    global mystate
    mystate = state
    keys = state.keys()
    return [{key: state[key][i:i+1].cpu().clone() for key in keys} for i in range(state[list(state.keys())[0]].shape[0])]

def runModel():
    while True:
        if len(thingsToDo) > 0:
            print("Concurrent:", thingsToDo.__len__(), flush=True, end='\r')
            thingsToDo2 = [thingsToDo.pop() for i in range(min(1024, len(thingsToDo)))]
            tokens = [[tok] for tok, state, do in thingsToDo2]
            
            state = mergestates([state for tok, state, do in thingsToDo2])

            out, state = model.forward(tokens, state)


            states = splitstates(state)

            [do(out[i], states[i]) for i, ( tok, state, do) in enumerate(thingsToDo2)]
            
        else:
            time.sleep(0.01)

async def addToStack(inp):
    tok, state = inp

    class backfix:
        def __init__(self, tok, state):
            self.tok = tok
            self.state = state
            self.done = False

        def setTok(self, logits, state):
            self.logits = logits
            self.state = state
            self.done = True
    
    infrequest = backfix(tok, state)

    do = lambda logits,newstate: infrequest.setTok(logits, newstate)

    
    thingsToDo.append((tok,state,do))

    while not infrequest.done:
        await asyncio.sleep(0.01)

    return infrequest.logits, infrequest.state

        
async def evaluate(
    prompt,
    model,
    pipeline,
    token_count=1024,
    temperature=0.8,
    top_p=0.8,
    presencePenalty = 0.5,
    countPenalty = 0.5,
    typicalSampling = True,
    state = None
):
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)),
                    top_p = float(top_p),
                    alpha_frequency = countPenalty,
                    alpha_presence = presencePenalty,
                    token_ban = [], # ban the generation of some tokens
                    token_stop = [0, 65532]) # stop generation whenever you see any token here

    ctx = prompt
    
    # gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    # print(f'vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')
    
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    for i in range(int(token_count)):
        if i == 0:
            out, state = model.forward([pipeline.encode(ctx)[-ctx_limit:]], state)
            out = out[0]
        else:
            promise = addToStack((token, state))
            out, state = await promise
            

        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)
            
        if typicalSampling:
            token = sample_logits_typical(out, temperature=args.temperature, top_p=args.top_p)
        else:
            token = sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        
        if token in args.token_stop:
            tmp = pipeline.decode(all_tokens[out_last:])
            yield tmp, state
            break
        all_tokens += [token]
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        
        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            out_str += tmp
            yield tmp, state
            out_last = i + 1




    
async def handleRWKVDirect(prompt, model, pipeline, params):
    typicalSampling = True
    
    statee = None
    
    # temperature = max(0.2, float(params['temperature']))
    temperature = max(0.2, float(params.get('temperature', 0.8)))
    top_p = float(params.get('top_p', 0.8))
    presencePenalty = float(params.get('presencePenalty', 0.5))
    countPenalty = float(params.get('countPenalty', 0.5))


    
    
    response = ""
    async for token, statee in evaluate(prompt, model, pipeline, typicalSampling=typicalSampling, state=statee, presencePenalty=presencePenalty, countPenalty=countPenalty, temperature=temperature, top_p=top_p):
        
        response += token

        for i in range(len(params.get("stop", []))):
            if params.get("stop", [])[i] in response:
                return

        yield token
        
        
        
        await asyncio.sleep(0.000001)
        

    
    print ("## Prompt ##")
    print (prompt)
    print ("## Response ##")
    print (response)
    
    print ("##################")
        
    # cacheKey = full_response.strip() + "<|im_end|>\n"
    # cachedStates[hash(cacheKey)] = (statee, time.time() + 60 * 60) # cache state for 1 hour
    gc.collect()
        

from aiohttp import web
import logging
import aiohttp

async def buildOutputChunk(token):
    object = {
        'object': 'chat.completion.chunk',
        'choices': [
            {
              "delta": {
                "content": token
              },
              "finish_reason": None,
              "index": 0
            }
        ],
    }
    return "data: " + json.dumps(object) + "\n\n"




async def handleCompletion(request):
    model, pipeline, index = await getModel()
    try:
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={'Content-Type': 'text/plain'},
        )
        await response.prepare(request)
        # get the request data (json)
        data = await request.json()    
        
        startTime = time.time()
        totalTokens = 0
        
        # run handleRwkv generator and output the result
        async for token in handleRWKVDirect(data['prompt'], model, pipeline, data):
            await response.write((await buildOutputChunk(token)).encode())
            await asyncio.sleep(0.000001)
            totalTokens += 1
            # if stop token is found, break
            if ("stop" in data and token in data["stop"]):
                print("## Stop token found ##")
                return response
            # if max tokens is set, break at max tokens
            if("max_tokens" in data and totalTokens > data["max_tokens"] ):   
                print("## Max tokens reached ##")
                return response

        await response.write("data: [DONE]\n\n".encode())
            
        print(f"## Time taken: {time.time() - startTime} ##")
        print(f"## Tokens generated: {totalTokens} ##")
        print(f"## Tokens per second: {totalTokens / (time.time() - startTime)} ##")

        
        await response.write_eof()
        return response
    except OSError:
        print("## Client disconnected ##")
               
def getModelM(request):
    #  id: model,
    #       object: "model",
    #       created: 1686935002,
    #       owned_by: "recursal.ai"
    return web.json_response({
        "data": [
            {
                "id": args.load_model.split(".")[0],
                "object": "model",
                "created": 1686935002,
                "owned_by": "recursal.ai"
            }
        ]
    })

app = web.Application()
logging.basicConfig(level=logging.DEBUG)
app.add_routes([
    web.post('/v1/completions', handleCompletion),
    web.get('/v1/models', getModelM),
])

threading.Thread(target=runModel).start()

web.run_app(app, port=PORT)

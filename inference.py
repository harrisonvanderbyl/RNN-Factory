########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import types, time, gc
import torch

from src.samplers import sample_logits

args = types.SimpleNamespace()

########################################################################################################
# Step 1: set model & config (use v4 to run your trained-from-scratch models. v4 and v4neo are compatible)
########################################################################################################




# MODEL_NAME = '/home/harrison/Documents/RNN-Factory/src/training/pipeline/models/5.pth'
MODEL_NAME = '/media/harrison/backup/7B.pth'
args.load_model = MODEL_NAME


from src.models.modules.Linear import InferenceLinear, Quantized, Linear

args.linear = Quantized

from src.models import RWKV_v4, RWKV_v5, Experimental

model = RWKV_v5(args)

from src.tokenizer import neox, world
tokenizer = world

context =   '''
### Instruction:
Tell me what a raven is
### Response:
'''

doGreedy = False

NUM_TRIALS = 999
LENGTH_PER_TRIAL = 333

TEMPERATURE = 0.9
top_p = 0.9
top_p_newline = 0.9  # only used in TOKEN_MODE = char

DEBUG_DEBUG = False  # True False --> show softmax output

########################################################################################################


model = model.eval()
model = model.requires_grad_(False)
# model = model.float()
model = model.float()
model = model.cuda()

# get model memory use
print("Memory use:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")





testdata = torch.randint(0, 100, (64,))

model.resetState()
atonce = model.forward(testdata, allLogits=True)
print(f'At once:', atonce.shape)
model.resetState()
# model = model.cuda()
for i in range(len(testdata)):
    atatime = model.forward(testdata[i:i+1])
    error = torch.max(torch.abs(atonce[0,i] - atatime)).item()
    # 3 decimal places
    error = int(error * 1000)
    print(f'[{i}]', error / 1000, 'max error')

########################################################################################################

# if tokenizer.charMode:
#     context = tokenizer.refine_context(context)
#     ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context]
# else:
ctx = tokenizer.encode(context)

src_len = len(ctx)
src_ctx = ctx.copy()

print("\nYour prompt has " + str(src_len) + " tokens.")
print(
    "Note: currently the first run takes a while if your prompt is long, as we are using RNN to preprocess the prompt. Use GPT to build the hidden state for better speed.\n"
)

time_slot = {}
time_ref = time.time_ns()

def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt

init_state = None
init_out = None
state = None
out = None

for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
    
    print(("-" * 50) + '\n' + context, end="")

    time_ref = time.time_ns()
    ctx = src_ctx.copy()

    if TRIAL == 0:
        
        gc.collect()
        torch.cuda.empty_cache()

    record_time('preprocess')
    out_last = src_len
    for i in range(src_len, src_len + (1 if DEBUG_DEBUG else LENGTH_PER_TRIAL)):
        x = ctx[: i + 1]
        x = x[-1:]

        if i == src_len:
            model.resetState()

            out = model.forward(ctx)
           
           
        else:
            out = model.forward(x)
        if DEBUG_DEBUG:
            print("model", np.array(x), "==>", np.array(out), np.max(out.cpu().numpy()), np.min(out.cpu().numpy()))
        # if TOKEN_MODE == "pile":
        # out[0] = -99  # disable <|endoftext|>
        if doGreedy:
            ttt = torch.argmax(out, dim=-1).item()
        else:
            ttt = sample_logits(
                out, temperature=TEMPERATURE, top_p=top_p
            )
        if ttt == 0:
            break
        ctx += [ttt]

        # if tokenizer.charMode:
        #     char = tokenizer.itos[ttt]
        #     print(char, end="", flush=True)
        # else:
        char = tokenizer.decode(ctx[out_last:])
        if '\ufffd' not in char: # is valid utf8 string?
            print(char, end="", flush=True)
            out_last = i+1

    record_time('total')
    # print(f'\n\n{time_slot}\n\n')
    print(
        f"\n\n--- preprocess {round(time_slot['preprocess'], 2)}s, generation {round(time_slot['total']-time_slot['preprocess'], 2)}s ", end = ''
    )

print(("-" * 50) + '\n')

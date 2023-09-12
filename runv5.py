########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import types, time, gc
import torch
from src.utils import TOKENIZER

args = types.SimpleNamespace()

########################################################################################################
# Step 1: set model & config (use v4 to run your trained-from-scratch models. v4 and v4neo are compatible)
########################################################################################################


TOKEN_MODE = "pile"
WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None

MODEL_NAME = '/home/harrison/Documents/RNN-Factory/out/rwkv-15.pth'

args.load_model = MODEL_NAME



context = "==An example of c++ code==\n"


NUM_TRIALS = 999
LENGTH_PER_TRIAL = 333

TEMPERATURE = 1.0
top_p = 0.8
top_p_newline = 0.9  # only used in TOKEN_MODE = char

DEBUG_DEBUG = False  # True False --> show softmax output

########################################################################################################

from src.model import RWKV

model = RWKV(args)
model = model.eval()
model = model.requires_grad_(False)
model = model.float()
# model = model.half()
model = model.cuda()

# get model memory use
print("Memory use:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")

# model = model.half()

print(f'\nOptimizing speed...')
model.forward([187])




print(f'\nLoading tokenizer {WORD_NAME}...')
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)
if TOKEN_MODE == "pile":
    assert tokenizer.tokenizer.decode([187]) == '\n'

testdata = "The Cute RNN model:"
testdata = tokenizer.tokenizer.encode(testdata)

model.resetState()
atonce = model.forward(testdata, allLogits=True)
print(f'At once:', atonce.shape)
model.resetState()
for i in range(len(testdata)):
    atatime = model.forward(testdata[i:i+1])
    error = torch.max(torch.abs(atonce[0,i] - atatime)).item()
    # 3 decimal places
    error = int(error * 1000)
    print(f'[{i}]', error / 1000, 'max error')

########################################################################################################

if tokenizer.charMode:
    context = tokenizer.refine_context(context)
    ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context]
else:
    ctx = tokenizer.tokenizer.encode(context)
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
        init_out, init_state = model.forward(ctx, returnState=True)
        gc.collect()
        torch.cuda.empty_cache()

    record_time('preprocess')
    out_last = src_len
    for i in range(src_len, src_len + (1 if DEBUG_DEBUG else LENGTH_PER_TRIAL)):
        x = ctx[: i + 1]
        x = x[-1:]

        if i == src_len:
            out = init_out.clone()
            model.setState(init_state)
        out = model.forward(x)
        if DEBUG_DEBUG:
            print("model", np.array(x), "==>", np.array(out), np.max(out.cpu().numpy()), np.min(out.cpu().numpy()))
        if TOKEN_MODE == "pile":
            out[0] = 0  # disable <|endoftext|>
  
        ttt = tokenizer.sample_logits(
            out,
            x,
            1,
            temperature=TEMPERATURE,
            top_p_usual=top_p,
            top_p_newline=top_p_newline,
        )
        ctx += [ttt]

        if tokenizer.charMode:
            char = tokenizer.itos[ttt]
            print(char, end="", flush=True)
        else:
            char = tokenizer.tokenizer.decode(ctx[out_last:])
            if '\ufffd' not in char: # is valid utf8 string?
                print(char, end="", flush=True)
                out_last = i+1

    record_time('total')
    # print(f'\n\n{time_slot}\n\n')
    print(
        f"\n\n--- preprocess {round(time_slot['preprocess'], 2)}s, generation {round(time_slot['total']-time_slot['preprocess'], 2)}s ", end = ''
    )

print(("-" * 50) + '\n')

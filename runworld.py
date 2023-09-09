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

class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[max(i,1)], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()

########################################################################################################
import torch.nn.functional as F
def sample_logits(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out.float().cpu(), dim=-1).numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out

########################################################################################################

tokenizer = RWKV_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

TOKEN_MODE = "pile"
WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None

MODEL_NAME = '/media/harrison/0F8D9B194C1273DC/rwkv-286.pth'

args.load_model = MODEL_NAME



context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in"


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
# model = model.half()
model = model.cuda()
model = model.to(torch.bfloat16)

# get model memory use
print("Memory use:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")

# model = model.half()

print(f'\nOptimizing speed...')
model.forward([187])


print(f'\nLoading tokenizer {WORD_NAME}...')

########################################################################################################

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
  
        ttt = sample_logits(
            out
        )
        ctx += [ttt]

        
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

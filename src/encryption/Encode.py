
import types
import torch

args = types.SimpleNamespace()
from safetensors import safe_open
from safetensors.torch import save_file

from src.tokenizer import world
tokenizer = world
MODEL_NAME = '/home/harrison/Documents/RNN-Factory/src/training/pipeline/models/5.pth'

args.load_model = MODEL_NAME

inpute = "message=[A red car with many people inside]"
context = f"### Instruction:\nRepeat the input exactly \n### Input:\n{inpute} \n### Response:"

########################################################################################################

from src.models import RWKV_v5

model = RWKV_v5(args)
model = model.eval()
model = model.requires_grad_(False)
# model = model.half()
model = model.to(torch.bfloat16)
model = model.cuda()

# get model memory use
print("Memory use:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")

# model = model.half()

print(f'\nOptimizing speed...')
model.forward([187])


########################################################################################################

# ctx = tokenizer.encode(context)

# model.forward(ctx)
# init_state = model.getState()
# for x in init_state.keys():
#     init_state[x] = init_state[x].to(torch.bfloat16)

# save_file(init_state, "secret.safetensors")

def encode(text):
    inpute = "message=[" + text + "]"
    context = f"### Instruction:\nRepeat the input exactly \n### Input:\n{inpute} \n### Response:"
    ctx = tokenizer.encode(context)
    model.resetState()
    model.forward(ctx)
    init_state = model.getState()
    # delete all values where key has "shift" in it
    for x in list(init_state.keys()):
        if "shift" in x:
            del init_state[x]
    
    # stack all values
    init_state = torch.stack(list(init_state.values()))

    # return init_state
    return init_state
    
def decode(state, max_len=100):
    # load state
    model.setState(state)
    # get output
    logits = model.forward([187])
    logits[0] = -99
    outputs = []

    while logits.argmax().item() != 0 and len(outputs) < max_len:
        outputs.append(logits.argmax().item())
        logits = model.forward([logits.argmax().item()])
        # print(logits.argmax())
    # return output
    try:
        return tokenizer.decode(outputs)
    except:
        return "Error: "
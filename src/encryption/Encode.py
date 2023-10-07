
import types
import torch

args = types.SimpleNamespace()
from safetensors import safe_open
from safetensors.torch import save_file

from src.tokenizer import world
tokenizer = world
MODEL_NAME = '/home/harrison/Documents/RNN-Factory/src/pipeline/models/5.pth'

args.load_model = MODEL_NAME

inpute = "message=[here is a secret message: |Entire Bee Movie Script|]"
context = f"### Instruction:\nRepeat the input exactly \n### Input:\n{inpute} \n### Response:"

########################################################################################################

from models import RWKV_v5

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

ctx = tokenizer.encode(context)

model.forward(ctx)
init_state = model.getState()
for x in init_state.keys():
    init_state[x] = init_state[x].to(torch.bfloat16)

save_file(init_state, "secret.safetensors")
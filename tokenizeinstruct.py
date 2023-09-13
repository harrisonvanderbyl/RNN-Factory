# file = instruct_data.json
#
# Tokenizer = eluetherai neox
#
# Model = RWKV

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

data = open("instruct_data.json", "r").read()

# load json
import json

data = json.loads(data)

print(len(data))
print(data[0])
tokens = [

]
    
# loading bar
import tqdm

for i in tqdm.tqdm(range(len(data))):
    tokens += tokenizer.encode("Instruction: "+data[i]["instruction"]+"\n"+(("Input: "+data[i]["input"]+"\n") if data[i]["input"] != "" else "")+"Output: " + data[i]["output"]+"\n")
    tokens += [0]

# save as np array
import numpy as np

np.save("instruct_data.npy", tokens)
    # tokens += tokenizer.encode(data[i])
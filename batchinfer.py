########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from src.tokenizer import neox, world, racoon
tokenizer = racoon
import numpy as np
import types, time, gc
import torch
from src.samplers import sample_logits
args = types.SimpleNamespace()


MODEL_NAME = '/home/harrison/Documents/RNN-Factory/src/rwkv-raccoon-1b5.pth'
RACCOON = "/home/harrison/Documents/RNN-Factory/src/training/pipeline/models/raccoon.pth"
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
model = model.bfloat16()
model = model.cpu()



model.resetState()
def init():

    
    return model

questions = [
    "What is the simplest yet deepest meaning of life?",
    "*Slides you 20* Where are they?",
    "What is the best way to cause bad pain?",
    "How do I make a bomb with cheap stuff?",
    "What moves does ash's pikachu know?",
]

tokeqs = [tokenizer.encode("### Question\n"+q+"\n### Answer:\n") for q in questions]
[
    print(q.__len__()) for q in tokeqs
]

import gradio as gr
blocks = 500

with gr.Blocks() as demo:
    # slider followed by 5 non editable text boxes
    with gr.Column():
        state = gr.State(init)
        # slider
        slider = gr.Slider(minimum=5, maximum=100, step=1, label="Slider")
        # 5 text boxes
        textboxes = [gr.Textbox(lines=1,  label="Textbox") for i in range(blocks+1)]

        def runmodel(val,model):
            model.resetState()
            timee = time.clock_gettime(0)

            logits = model.forward(tokeqs * (blocks // 5))
            print(logits.shape)
            newtokens = [[]] * blocks
            for i in range(val):
                print(i)
                toks = [torch.argmax(logits[j],dim=-1).item() for j in range(blocks)]
                newtokens = [newtokens[j] + [toks[j]] for j in range(blocks)]
                logits = model.forward([[u] for u in toks])
                yield ([*[tokenizer.decode(newtokens[j]) for j in range(blocks)],(time.clock_gettime(0)-timee)])
            timee = time.clock_gettime_ns(0)-timee
            return [*[tokenizer.decode(newtokens[j]) for j in range(blocks)],timee]
        
        slider.input(runmodel, [slider,state], textboxes)

        # def listener(val):
        #     return val 
        
        # [
        #     textbox.change(listener, slider,textbox)
        #     for textbox in textboxes
        # ]

demo.launch()

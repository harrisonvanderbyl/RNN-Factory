########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info

class RandomMemory(Dataset):
    def __init__(self, args):
        self.args = args
        self.vocab_size = args.vocab_size

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        ctx_len = args.ctx_len#random.randint(32, args.ctx_len//2)*2

        # create ctx_len//2 random integers in [0, vocab_size)
        splits = [2,4,8,16,32,64,128][random.randint(0,6)]
        splits = 2
        splitctx = (ctx_len//2) // splits
        dix = [[random.randint(1, self.vocab_size - 1) for _ in range(splitctx)]*2 for _ in range(splits)]

        x = torch.tensor(dix[:-1], dtype=torch.long).flatten()
        y = torch.tensor(dix[1:], dtype=torch.long).flatten()
        # z = torch.tensor((ctx_len//2)*[0]+(ctx_len//2 - 1)*[1], dtype=torch.bfloat16)

        return x, y

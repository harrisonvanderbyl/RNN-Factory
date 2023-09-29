########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime
from datasets import load_dataset
enwiki = load_dataset("teven/enwiki_100k", streaming=True, split="train") 
mcode = load_dataset("codeparrot/github-code", streaming=True, split="train",languages=["JavaScript","C++"])
instruct = load_dataset("WizardLM/WizardLM_evol_instruct_70k", streaming=True, split="train") 
mdata = iter(enwiki), iter(mcode), iter(instruct)

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        if args.data_type == "binidx":
            self.vocab_size = args.vocab_size
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

            if args.my_pile_version == 1:
                self.data = MMapIndexedDataset(args.data_file)
                self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
                rank_zero_info(f"Data has {self.data_size} tokens.")
            elif args.my_pile_version == 2:
                data_list = open(args.data_file, "r", encoding='utf-8').read().strip().split('\n')
                data_list = [i.strip().split(' ') for i in data_list]
                self.data = []
                self.data_size = int(data_list[-1][-1])
                rank_zero_info(f"Data has {self.data_size} chunks.")
                for d in data_list:
                    data = MMapIndexedDataset(d[0])
                    data_size = len(data._bin_buffer) // data._index._dtype_size
                    assert (data_size - args.ctx_len) == int(d[1])
                    self.data += [[int(d[-1]), int(d[1]), data]]
                # rank_zero_info(self.data)

            if args.my_qa_mask > 0:
                # self.data_pile = MMapIndexedDataset('/fsx/pile/pile_20B_tokenizer_text_document')
                self.data_pile = MMapIndexedDataset('/fsx/pile_deduped/pile_0.87_deduped_text_document')
                self.data_pile_size = len(self.data_pile._bin_buffer) // self.data._index._dtype_size
            else:
                self.data_pile = None
                self.data_pile_size = 0

            if args.my_pile_stage > 0:
                # assert self.data_size == 332115325534 and self.vocab_size == 50277
                self.samples_per_epoch = args.epoch_steps * args.real_bsz
                assert self.samples_per_epoch == 40320
                rank_zero_info(f"########## Pile 20b-tokenized stage {args.my_pile_stage} ##########")
                dataset_slot = self.data_size // args.ctx_len
                if args.my_pile_stage != 4:
                    assert MaybeIsPrime(args.magic_prime)
                    assert args.magic_prime % 3 == 2
                    assert args.magic_prime / dataset_slot > 0.99 and args.magic_prime / dataset_slot <= 1
        elif args.data_type == "numpy":
            self.data = np.load(args.data_file).astype("int")
            self.vocab_size = args.vocab_size
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            self.data_size = len(self.data)
            rank_zero_info(f"Data has {self.data_size} tokens.")
        elif args.data_type == "uint16":
            self.data = np.fromfile(args.data_file, dtype=np.uint16).astype("int32").reshape(-1, args.my_sample_len)
            self.vocab_size = args.vocab_size
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            self.data_size = self.data.shape[0]
            rank_zero_info(f"Data has {self.data_size} samples.")
        elif args.data_type == "wds_img":
            self.vocab_size = -1
            self.data_size = -1
            
            
            self.error_count = 0
        elif args.data_type == "stream":
            self.vocab_size = 50277
            self.data_size = -1
            
            self.data = mdata
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

            self.tokenizer = tokenizer
            self.error_count = 0
        else:
            if args.data_type == "dummy":
                rank_zero_info("Building dummy data...")
                self.data = ""
                for i in range(100000):
                    aa = (i) % 10000
                    bb = (i * i) % 10000
                    cc = aa + bb
                    self.data += f".{aa}+{bb}={cc}."
            else:
                self.data = open(args.data_file, "r", encoding=args.data_type).read()
            rank_zero_info("Building token list...")
            unique = sorted(list(set(self.data)))
            self.vocab_size = len(unique)
            # rank_zero_info()
            # for u in unique:
            #     print(u, end=' ')
            # rank_zero_info('\n\n')
            xx = 0
            xxObj = {}
            for u in unique:
                xxObj[xx] = u
                xx += 1
            with open(f"{args.proj_dir}/vocab.json", "w", encoding="utf-16le") as vocab_file:
                vocab_file.write(json.dumps(xxObj, ensure_ascii=False))
            self.data_size = len(self.data)
            rank_zero_info(f"Data has {self.data_size} tokens, {self.vocab_size} vocab size.")
            self.stoi = {ch: i for i, ch in enumerate(unique)}
            self.itos = {i: ch for i, ch in enumerate(unique)}

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")
        if args.data_type == "stream":
            
            rcode = []
            while len(rcode) < args.ctx_len:
            # if True:
                # choose between "enwiki", "code", "instruct"
                choices = ["enwiki","code","instruct"]
                # filter based on data_file, eg "" is all, "enwiki" is only enwiki and "code,instruct" is only code and instruct
                if args.data_file != "":
                    choices = args.data_file.split(",")
                choice = random.choice(choices)

                # try:
                if choice == "enwiki":
                
                    dd = next(self.data[0])
                    while len(dd["text"]) == 0:
                        dd = next(self.data[0])
                    # wiki = "Instruction: Tell me about "+dd["metadata"]["document_url"].replace("https://en.wikipedia.org/wiki/","").replace("_"," ")+"\nOutput: " +dd["text"]
                    
                    tokenized = self.tokenizer.encode(dd["text"])
                    rcode = rcode + [0] + tokenized
               
                if choice == "code":
                    dd = next(self.data[1])
                    code = dd["code"]
                    tokenized = self.tokenizer.encode(code)
                    rcode = rcode + [0] + tokenized
                
                if choice == "instruct":
                    dd = next(self.data[2])
                    instruct = "User: "+dd["instruction"]+"\nBot: " +dd["output"]
                    tokenized = self.tokenizer.encode(instruct)
                    rcode = rcode + [0] + tokenized
               
            tokenized = rcode[:args.ctx_len]
            # print("len", len(tokenized))
            
            
            dix = torch.tensor(tokenized, dtype=torch.long)
            # positions = torch.arange(0, len(dix), dtype=torch.long)
            # dix = torch.cat((dix.unsqueeze(1), positions.unsqueeze(1)), dim=-1)
            # shuffleindex = torch.randperm(len(dix[0]))
            # dix 

            x = dix[:-1]
            y = dix[1:]
            return x, y

        if args.data_type == "wds_img":
            def init_wds(self, bias=0):
                def identity(x):
                    return x            
                import webdataset as wds
                import torchvision.transforms as transforms
                # img_transform = transforms.Compose(
                #     [transforms.CenterCrop(256)]
                # )
                img_transform = transforms.Compose([
                    transforms.CenterCrop(512),
                    transforms.Resize((args.my_img_size))
                ])
                self.data_raw = wds.WebDataset(args.data_file, resampled=True).shuffle(10000, initial=1000, rng=random.Random(epoch*100000+rank+bias*1e9)).decode("torchrgb").to_tuple("jpg", "json", "txt").map_tuple(img_transform, identity, identity)
                for pp in self.data_raw.pipeline:
                    if 'Resampled' in str(pp):
                        pp.deterministic = True
                        def worker_seed():
                            return rank*100000+epoch+bias*1e9
                        pp.worker_seed = worker_seed
                self.data = iter(self.data_raw)
                # print(f"WebDataset loaded for rank {rank} epoch {epoch}")
            if self.data == None:
                init_wds(self)
            trial = 0
            while trial < 10:
                try:
                    dd = next(self.data) # jpg, json, txt
                    break
                except:
                    print(f'[dataloader error - epoch {epoch} rank {rank} - trying a new shuffle]')
                    self.error_count += 1
                    init_wds(self, self.error_count)
                    trial += 1
                    pass
            # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} {dd[2]}")
            # with open(f"sample_{rank}.txt", "a", encoding="utf-8") as tmp:
            #     tmp.write(f"epoch {epoch} idx {idx} rank {rank}/{world_size} {int(dd[1]['key'])}\n")
            return dd[0], dd[2]
        else:
            if args.data_type == "uint16":
                i = np.random.randint(0, self.data_size-1)
                dix = self.data[i]
                x = torch.tensor(dix[:-1], dtype=torch.long)
                y = torch.tensor(dix[1:], dtype=torch.long)
            else:
                ctx_len = args.ctx_len
                req_len = ctx_len + 1
                magic_prime = args.magic_prime
                data = self.data

                if args.my_pile_stage > 0:
                    ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

                    if args.my_qa_mask > 0:
                        ii_orig = ii
                        if ii % 2 == 0:
                            ii = -1
                            data = self.data_pile
                        else:
                            ii = ii // 2
                    if data == self.data_pile:
                        i = np.random.randint(0, self.data_pile_size - req_len)
                    else:
                        if args.my_pile_stage == 4 or ii < args.my_random_steps:
                            # cheat: pick a random spot in dataset
                            if args.my_pile_version == 1:
                                i = np.random.randint(0, self.data_size - req_len)
                            else:
                                i = np.random.randint(0, self.data_size)
                        else:
                            ii = ii - args.my_random_steps
                            factor = (math.sqrt(5) - 1) / 2
                            factor = int(magic_prime * factor)
                            i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
                            i = i + args.my_pile_shift
                    # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")
                else:
                    # cheat: pick a random spot in dataset
                    i = np.random.randint(0, self.data_size - req_len)

                if args.data_type == "binidx":
                    if args.my_pile_version == 1:
                        dix = data.get(idx=0, offset=i, length=req_len).astype(int)
                    else:
                        # self.data : cutoff, chunk_count, data
                        for j in range(len(data)):
                            if i < data[j][0]:
                                ii = i
                                i = (i - (data[j-1][0] if j > 0 else 0)) % data[j][1]
                                dix = data[j][2].get(idx=0, offset=i, length=req_len).astype(int)
                                # print(ii, j, i)
                                break
                elif args.data_type == "numpy":
                    dix = data[i : i + req_len]
                else:
                    dix = [self.stoi[s] for s in data[i : i + req_len]]

                if args.my_qa_mask == 1:
                    if data == self.data_pile:
                        z = [1] * ctx_len
                    else:
                        z = [0] * ctx_len
                        z_sum = 0
                        isGood = False
                        for i in range(3, ctx_len):
                            if dix[i] == 27 and dix[i-1] == 34 and dix[i-2] == 187 and dix[i-3] == 187:
                                isGood = True
                            if dix[i] == 0:
                                isGood = False
                            if isGood:
                                z[i] = 1
                                z_sum += 1
                        if z_sum == 0:
                            z = [1] * ctx_len
                            i = np.random.randint(0, self.data_pile_size - req_len)
                            dix = self.data_pile.get(idx=0, offset=i, length=req_len).astype(int)
                    z = torch.tensor(z, dtype=torch.bfloat16)

                x = torch.tensor(dix[:-1], dtype=torch.long)
                y = torch.tensor(dix[1:], dtype=torch.long)

                # if ii_orig < 50:
                #     # if rank == 1:
                #     print('rank', rank, 'i', ii_orig, ii, i, 'x', x[:5], '...', x[-5:])
                # else:
                #     exit(0)

                if args.my_qa_mask == 1:
                    return x, y, z

            return x, y

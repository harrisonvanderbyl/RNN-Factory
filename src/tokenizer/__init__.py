from .tokenizer import RWKV_TOKENIZER, neox
from .raccoon import RaccTRIE_TOKENIZER

fname = "rwkv_vocab_v20230424.txt"
world = RWKV_TOKENIZER(__file__[:__file__.rindex('/')] + '/' + fname)

neox = neox

racfname = "rwkv_vocab_v20230922_chatml.txt"
racoon = RaccTRIE_TOKENIZER(__file__[:__file__.rindex('/')] + '/' + racfname)
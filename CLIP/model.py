import io
import os
import ctypes
import struct
import argparse
import numpy as np
from typing import List, Tuple, Dict
import ggml
from ggml.experimental import GGML_FTYPE, Context, InitParams, Tensor, GGML_TYPE, CGraph

def compute_ctx_size(fin: io.BufferedReader) -> int:
    # Save current position in file and get file size, then return
    position = fin.tell()
    ctx_size = 0
    while True:
        nbytes = struct.calcsize("iii")
        data = fin.read(nbytes)
        if len(data) != data:
            break
        (n_dims, s_len, ftype) = struct.unpack("iii", data)
        dims = struct.unpack("i"*n_dims, fin.read(struct.calcsize("i"*n_dims)))
        match ftype:
            case 0:
                _format = "f"
            case 1:
                _format = "e"
        n_bytes = struct.calcsize(_format * int(np.prod(dims)))
        ctx_size += n_bytes
        ctx_size += 256 
        name = fin.read(s_len).decode("utf-8")
        fin.seek(n_bytes, os.SEEK_CUR)

    fin.seek(position)

    return ctx_size

class ResidualAttentionBlock:
    def __init__(self, ctx:Context, wtype:GGML_FTYPE, embed_dim:int, heads:int, use_attention_mask: bool=False) -> None:
        self.tensors: Dict[str, Tensor] = {}
        self.n_head = heads
        self.embed_dim = embed_dim
        self.use_attention_mask = use_attention_mask


        # Multihead Attention 
        self.in_proj_weight = Tensor.new_tensor_2d(wtype, embed_dim, 3*embed_dim, ctx=ctx)
        
        # Layer Norm 1 (ln_1)
        self.ln_1_weight = Tensor.new_tensor_1d(wtype, embed_dim, ctx=ctx)
        self.ln_1_bias = Tensor.new_tensor_1d(wtype,embed_dim, ctx=ctx)
        self.tensors["ln_1.weight"] = self.ln_1_weight
        self.tensors["ln_1.bias"] = self.ln_1_bias





        


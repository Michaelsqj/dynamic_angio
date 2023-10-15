"""
Code to define linear operators and concatenate with the autoencoder
"""

import torch
import sigpy as sp
import numpy as np
import cupy as cp
from tqdm import tqdm
import random

# 1. create the linops for NUFFT and coil sensitivity maps for each time point and coil
class PFCx:
    def __init__(self, coords, mps, Nt_sub, Nc_sub, repr_str=None):
        # coords: [Nt, pts_shape, ndim]
        # mps: [Nc, im_size]
        # Nt_sub: number of time points to select
        # Nc_sub: number of coils to select
        self.coords = coords
        self.mps = mps
        self.Nt_sub = Nt_sub
        self.Nc_sub = Nc_sub

        self.nt = coords.shape[0]
        self.pts_shape = coords.shape[1:-1]
        self.ndim = coords.shape[-1]
        self.ncoils = mps.shape[0]
        self.im_size = list(mps.shape[1:])

        self.ishape = [self.nt] + list(self.im_size)
        self.oshape = [Nt_sub, Nc_sub] + list(self.im_size)

        self.nufft_list = [] 
        for i in tqdm(range(self.nt)):
            self.nufft_list.append(sp.linop.NUFFT(ishape=[1, Nc_sub]+self.im_size, coord=coords[i,...]))
        
        self.coils = list(range(self.ncoils))
        self.times = list(range(self.nt))

    def apply(self, input):
        # input: [Nt, im_size, 2]
        # output: [Nt_sub, Nc_sub, im_size, 2]
        
        # 1. select time points and coils
        sub_coils, sub_times = self.select_subsets()
        print(f"sub_coils: {sub_coils}, sub_times: {sub_times}")

        # 2. create the linops of coils
        mps_ops = sp.linop.Multiply(shape=[self.Nt_sub,1]+list(self.im_size), mps=self.mps[sub_coils,...])

        # 3. create the linops of NUFFT
        nufft_ops = sp.linop.Diag([self.nufft_list[i] for i in sub_times], iaxis=0,oaxis=0)

        # 4. combined the ops
        ops = nufft_ops * mps_ops

        # 5. convert the ops to pytorch function
        ops_torch = sp.to_pytorch_function(ops, input_iscomplex=True, output_iscomplex=True)

        # 6. apply the ops
        output = ops_torch.apply(input)

        # Empty the cache to free up GPU memory
        torch.cuda.empty_cache()

        return output

    def select_subsets(self):
        random.shuffle(self.coils)
        random.shuffle(self.times)
        sub_coils = sorted(self.coils[:self.Nc_sub])
        sub_times = sorted(self.times[:self.Nt_sub])
        return sub_coils, sub_times

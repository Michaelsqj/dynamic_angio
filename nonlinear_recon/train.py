import torch
import sigpy as sp
from sigpy.mri import pipe_menon_dcf
import numpy as np
import cupy as cp
from tqdm import tqdm
import random
import models
import os
from ae_linops import PFCx
import scipy.io as sio
import mat73
import time
import nibabel as nib
from make_phantom import create_phantom

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt


"""
alpha is like the only trainbale parameter in the network
sub_coils and sub_times are the input to the network
"""
class Trainer():
    def __init__(self, coords, mps, inputs) -> None:
        # coords: [Nt, pts_shape, ndim]
        # mps: [Nc, im_size]
        # inputs: (Nt,  Nc, pts_shape)
        self.coords = coords
        self.mps = mps
        self.inputs = torch.stack([torch.from_numpy(inputs.real), torch.from_numpy(inputs.imag)], dim=-1).double() # (Nt,  Nc, pts_shape, 2)
        print(f"inputs shape {self.inputs.shape}")
        self.im_size = list(mps.shape[1:])
        self.pts_shape = list(coords.shape[1:-1])
        self.Nt = coords.shape[0]
        self.Nc = mps.shape[0]

        # Define the parameters
        T=self.Nt
        self.latent_num=3
        model_layers=2
        nonlinearity='tanh'

        self.Nt_sub = 144
        self.Nc_sub = 8

        self.coils = list(range(self.Nc))
        self.times = list(range(self.Nt))

        # calculate dcf or preconditioner
        print("calculate dcf or preconditioner")
        self.sqrt_dcf = self.calc_dcf()
        self.sqrt_dcf_torch = torch.stack([torch.from_numpy(self.sqrt_dcf.real), torch.from_numpy(self.sqrt_dcf.imag)], dim=-1).double() # [Nt, pts_shape, 2]
        # self.sqrt_dcf = np.ones(self.coords.shape[:2])

        # intialize alphas, [prod(im_size), latent_num, 2]
        # initialize alphas using gridded reconstruction
        # print("initialize alphas using gridded reconstruction")
        # rd = self.grid_recon()

        # initialize alphas with random values
        print("initialize alphas with random values")
        self.alphas = torch.zeros(self.im_size + [self.latent_num, 2], requires_grad=False)

        # load the ae model
        # construct the model
        print(f"load model {T}_{self.latent_num}_{model_layers}_{nonlinearity}")
        self.model = models.autoencoder(T,self.latent_num,model_layers,nonlinearity)
        # load the model parameters
        self.model.load_state_dict(torch.load(os.path.join('checkpoints', f"{T}_{self.latent_num}_{model_layers}_{nonlinearity}")))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # intialize alphas, [prod(im_size), latent_num, 2]
        # initialize alphas using gridded reconstruction
        ops = self.create_linops(self.coils, self.times, return_pytorch=False)
        opsH = sp.to_pytorch_function(ops.H, input_iscomplex=True, output_iscomplex=True)
        rd = opsH.apply((torch.unsqueeze(self.sqrt_dcf_torch, 1)*self.inputs).to('cuda')) #[Nt_sub, im_size, 2]
        rd = torch.permute(torch.reshape(rd, [self.Nt, np.prod(self.im_size)*2]), [1,0])
        print(rd.dtype)
        self.alphas = torch.permute(torch.reshape(self.model.to('cuda').encode(rd.to(torch.float32)), self.im_size+[2,self.latent_num]), [0,1,2,4,3]).detach().cpu().requires_grad_(False)


        # set up the optimizer SGD
        print("set up the optimizer SGD")
        # self.optimizer = torch.optim.Adam([self.alphas], lr=1e0)
        # self.optimizer = torch.optim.SGD([self.alphas], lr=1e7, momentum=0.9)
        self.step_size = 1e6

        # set up the loss function
        self.mse_loss = torch.nn.MSELoss()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.writer = SummaryWriter("/well/okell/users/dcs094/data/dynamic_recon/runs/exp-small_phantom")

    def forward(self, sub_coils, sub_times):
        Nt_sub = len(sub_times)
        Nc_sub = len(sub_coils)
        # alphas: [im_size, latent_num, 2]
        x = torch.reshape(self.alphas, [np.prod(self.im_size),self.latent_num, 2])
        x = torch.permute(x, [0,2,1])
        x = torch.reshape(x, [np.prod(self.im_size)*2, self.latent_num])
        # 1. pass the alphas to the autoencoder
        #       output x: [prod(im_size)*2, Nt]
        x = self.model.decode(x)[:,sub_times]

        x = torch.permute(x, [1,0])
        x = torch.reshape(x, [Nt_sub]+self.im_size+[2]) # [Nt_sub, im_size, 2]

        # 2. create linops for NUFFT and coil sensitivity maps
        linops_torch = self.create_linops(sub_coils, sub_times)
        

        # 3. pass the imgs to PFCx operator
        #       output kd: [nt, ncoils, npts, 2]
        x = linops_torch.apply(x)

        return x

    def create_linops(self, sub_coils, sub_times, return_pytorch=True):
        # input: [Nt_sub, im_size, 2]
        # output: [Nt_sub, Nc_sub, pts_shape, 2]
        Nc_sub = len(sub_coils)
        Nt_sub = len(sub_times)
        nufft_list = [] 
        for i in range(self.Nt):
            nufft_list.append(sp.linop.NUFFT(ishape=[1, Nc_sub]+self.im_size, coord=self.coords[i,...]))

        reshape_ops = sp.linop.Reshape(ishape=[Nt_sub]+self.im_size, oshape=[Nt_sub,1]+self.im_size)

        # 2. create the linops of coils
        mps_ops = sp.linop.Multiply(ishape=[Nt_sub,1]+self.im_size, mult=self.mps[sub_coils,...])

        # 3. create the linops of NUFFT
        nufft_ops = sp.linop.Diag([nufft_list[i] for i in sub_times], iaxis=0,oaxis=0)

        # 4. create dcf ops
        dcf_ops = sp.linop.Multiply(ishape=nufft_ops.oshape, mult=self.sqrt_dcf[sub_times,np.newaxis,:])

        # 5. combined the ops
        ops = dcf_ops*nufft_ops * mps_ops * reshape_ops

        if return_pytorch:
            return sp.to_pytorch_function(ops, input_iscomplex=True, output_iscomplex=True)
        else:
            return ops

    def calc_dcf(self, dev_id=0):
        dev = sp.Device(dev_id)
        xp = dev.xp
        start = time.time()
        sqrt_dcf = np.zeros(self.coords.shape[:2])
        with dev:
            for i in tqdm(range(self.Nt)):
                dcf = pipe_menon_dcf(self.coords[i,...], img_shape=self.im_size, device=dev,show_pbar=False).real.astype(xp.float32)
                dcf /= xp.linalg.norm(dcf.ravel(), ord=xp.inf)
                sqrt_dcf[i,:] = sp.to_device(xp.sqrt(dcf),-1)
                torch.cuda.empty_cache()
        end = time.time()
        print(f"dcf took {end-start} s")
        print(f"dcf shape {dcf.shape}")
        return sqrt_dcf

    def select_subsets(self):
        # random.shuffle(self.coils)
        # random.shuffle(self.times)
        # sub_coils = sorted(self.coils[:self.Nc_sub])
        # sub_times = sorted(self.times[:self.Nt_sub])
        dt = int(np.floor(self.Nt/self.Nt_sub))
        dc = int(np.floor(self.Nc/self.Nc_sub))
        sub_coils = sorted(self.coils[::dc][:self.Nc_sub])
        sub_times = sorted(self.times[::dt][:self.Nt_sub])
        
        return sub_coils, sub_times
    
    def train(self, device='cuda', epochs=1000):
        self.alphas = self.alphas.to(device)
        self.alphas.requires_grad = True
        self.model = self.model.to(device)
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            # self.optimizer.zero_grad()
            if self.alphas.grad is not None:
                self.alphas.grad.zero_()
            sub_coils, sub_times = self.select_subsets()
            # 1. forward, calculate loss
            x = self.forward(sub_coils, sub_times)  # output: [Nt_sub, Nc_sub, pts_shape, 2]
            loss = self.mse_loss(x, self.inputs[sub_times, ...][:,sub_coils,...].to(device))
            # 2. backward
            loss.backward()

            # 4. print loss
            with torch.no_grad():
                # print(f"alphas before: {self.alphas.flatten()[10000:10010]}")
                # print(f"alphas grad : {self.alphas.grad.flatten()[10000:10010]}")
                pbar.set_description(f"epoch {epoch}, loss {loss.item()}, {torch.sum(self.alphas.grad)}")

            # 3. update parameters
            # self.optimizer.step()
            with torch.no_grad():
                self.alphas -= self.step_size * self.alphas.grad
            # with torch.no_grad():
            #     print(f"alphas after: {self.alphas.flatten()[10000:10010]}")
            # 5. log
            self.log(epoch, loss.item())
            # with open('loss.txt', 'a') as f:
            #     f.write(f"{loss.item()}\n")
            torch.cuda.empty_cache()

        self.writer.close()

    def log(self, epoch, loss):
        self.writer.add_scalar('Loss/train', loss, epoch)
        xx, xy, xz = self.mip()
        self.writer.add_image('MIP/x', xx, epoch)
        self.writer.add_image('MIP/y', xy, epoch)
        self.writer.add_image('MIP/z', xz, epoch)

    def mip(self):
        # max intensity projection of self.alphas
        # output: [im_size, 2]
        with torch.no_grad():
            x = (self.alphas.detach().cpu().numpy()**2).sum(axis=-1)**0.5
            xx = np.max(x, axis=0)
            xx = np.reshape(xx, [xx.shape[0],-1], order='F')
            xx = ((xx-xx.min())/(xx.max()-xx.min()) * 255).astype(np.uint8)
            xx = np.stack([xx,xx,xx], axis=0)
            xy = np.max(x, axis=1)
            xy = np.reshape(xy, [xy.shape[0],-1], order='F')
            xy = ((xy-xy.min())/(xy.max()-xy.min()) * 255).astype(np.uint8)
            xy = np.stack([xy,xy,xy], axis=0)
            xz = np.max(x, axis=2)
            xz = np.reshape(xz, [xz.shape[0],-1], order='F')
            xz = ((xz-xz.min())/(xz.max()-xz.min()) * 255).astype(np.uint8)
            xz = np.stack([xz,xz,xz], axis=0)
            return xx, xy, xz

    
    # def grid_recon(self, dev_id=0):
    #     outputs = np.zeros([self.Nt,self.Nc]+self.im_size, dtype=complex)
    #     dev = sp.Device(dev_id)
    #     xp = dev.xp
    #     # define NUFFT adjoint operator for each frame
    #     print("define NUFFT adjoint operator for each frame")
    #     nufft_adj_list = [] 
    #     for i in tqdm(range(self.Nt)):
    #         nufft_adj_list.append(sp.linop.NUFFTAdjoint(oshape=[self.Nc]+self.im_size,coord=self.coords[i,...]))
    #     # calculate the outputs
    #     print("calculate adjoint NUFFT outputs")
    #     with dev:
    #         for i in tqdm(range(self.Nt)):
    #             outputs[i,...] = sp.to_device(nufft_adj_list[i](sp.to_device(self.sqrt_dcf[i,...]*inputs[i,...],0)), -1)
    #             torch.cuda.empty_cache()
    #     rd = np.zeros([self.Nt]+self.im_size, dtype=complex)
    #     print("romer coil combination")
    #     for i in tqdm(range(self.Nt)):
    #         rd[i,...] = np.sum(outputs[i,...]*np.conjugate(self.mps),axis=0) / np.sum(np.abs(self.mps)**2,0)
    #     return rd



if __name__ ==  '__main__':
    # # load the data
    # ktraj = sio.loadmat('/well/okell/users/dcs094/data/subspace/rawdata/raw_data_1-12-22/ktraj.mat')['ktraj']
    # kdata = sio.loadmat('/well/okell/users/dcs094/data/subspace/rawdata/raw_data_1-12-22/kdata.mat')['kdata']
    # param = sio.loadmat('/well/okell/users/dcs094/data/subspace/rawdata/raw_data_1-12-22/param.mat')['param']
    # print(ktraj.shape, kdata.shape)
    # print(param)
    # # load coil sensitivity
    # sens = mat73.loadmat('/well/okell/users/dcs094/data/subspace/rawdata/raw_data_1-12-22/meas_MID00169_FID00171_qijia_CV_VEPCASL_halfg_johnson_60_1_3_500_24x48_100hz_176_vFA_sens1.mat')['sens']
    # sens = np.transpose(sens, (3,0,1,2))
    # # sens: (Nc, im_size)
    # print(sens.shape)

    # # 1. reshape ktraj, kdata to 144 frames
    # ## reformat the ktraj and kdata
    # im_size = [186, 196, 150]
    # # ktraj: (1398, 144, 48, 3)
    # nreadouts, nt, nshots, ndim = ktraj.shape

    # # kdata: (1398, 144, 48, 2, 8)
    # ncoils = kdata.shape[-1]
    # nframes = 144
    # nsegs = 1
    # # coords: (nframes,  nshots * nsegs * nreadouts , ndim)
    # # inputs: (nframes,  ncoils, nshots * nsegs * nreadouts)
    # coords = np.transpose(np.reshape(ktraj, (nreadouts, nframes, nsegs, nshots, ndim)), (1,3,2,0,4))
    # inputs = np.transpose(np.reshape(kdata[:,:,:,1,:]-kdata[:,:,:,0,:], (nreadouts, nframes, nsegs, nshots, ncoils)), (1,4,3,2,0))
    # coords = np.reshape(coords, (nframes,  nshots * nsegs * nreadouts , ndim)) * np.array(im_size)/2 / np.pi
    # inputs = np.reshape(inputs, (nframes,  ncoils, nshots * nsegs * nreadouts))

    # print(f"inputs shape {inputs.shape}")
    # print(f"coords shape {coords.shape}")


    # # create phantom
    inputs, coords, sens = create_phantom()
    # set up the trainer
    print("set up the trainer")
    trainer = Trainer(coords=coords, mps=sens, inputs=inputs)
    trainer.train()
    # rd = np.abs(trainer.rd)
    # # 6. save to nib file
    # print("save to nib file")
    # fpath = '/well/okell/users/dcs094/data/sigpy_test/tmp'
    # img = nib.Nifti1Image(np.transpose(rd,(1,2,3,0)), np.eye(4))
    # nib.save(img, fpath)
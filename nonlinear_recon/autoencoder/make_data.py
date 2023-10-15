import torch
import sigpy as sp
import numpy as np
import cupy as cp
from tqdm import tqdm
import random
import nibabel as nib
from sigpy.mri import radial, pipe_menon_dcf
import mat73
import models
import os
import scipy.io as sio

def create_phantom():
    dev = sp.Device(0)
    xp = dev.xp
    # 1. load the phantom
    fpath = "/well/okell/users/dcs094/data/subspace/modelfit144_small.nii.gz"
    nii_img = nib.load(fpath)
    img = nii_img.get_fdata()
    im_size = list(img.shape[:3]) # [62, 66, 50]
    nt = img.shape[-1]

    # 2. pass the image through the autoencoder
    phi = sio.loadmat("../data/basis_144.mat")['V'][:,:4]
    print(phi.shape)
    img = np.reshape(np.reshape(img,[-1,nt])@phi@phi.T, im_size+[nt])
    img = np.transpose(img, (3,0,1,2))

    # 2. create sampling trajectory
    nshots = 48*1
    coord=radial(coord_shape=(nt*nshots,128,3), img_shape=im_size).reshape([nt,-1,3])
    print(f"coord shape {coord.shape}")

    # 3. load coil sensitivity
    mps = mat73.loadmat('/well/okell/users/dcs094/data/dynamic_recon/sens.mat')['sens']
    print(f"mps shape {mps.shape}")

    Nc = mps.shape[0]
    Nt = coord.shape[0]

    nufft_list = [] 
    for i in tqdm(range(nt)):
        nufft_list.append(sp.linop.NUFFT(ishape=[1, Nc]+im_size, coord=coord[i,...]))

    reshape_ops = sp.linop.Reshape(ishape=[Nt]+im_size, oshape=[Nt,1]+im_size)

    # 2. create the linops of coils
    mps_ops = sp.linop.Multiply(ishape=[Nt,1]+im_size, mult=mps)

    # 3. create the linops of NUFFT
    nufft_ops = sp.linop.Diag(nufft_list, iaxis=0,oaxis=0)

    ops = nufft_ops * mps_ops * reshape_ops

    kdata = sp.to_device(ops(sp.to_device(img, device=dev)), device=-1)
    print(f"kdata shape {kdata.shape}")
    print(f"coords shape {coord.shape}")
    print(f"mps shape {mps.shape}")

    return kdata, coord, mps

def load_model():
    # Define the parameters
    T=144
    latent_num=3
    model_layers=2
    nonlinearity='tanh'
    # load the ae model
    # construct the model
    print(f"load model {T}_{latent_num}_{model_layers}_{nonlinearity}")
    model = models.autoencoder(T,latent_num,model_layers,nonlinearity)
    # load the model parameters
    model.load_state_dict(torch.load(os.path.join('checkpoints', f"{T}_{latent_num}_{model_layers}_{nonlinearity}")))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model

def load_invivo():
    # load the data
    ktraj = sio.loadmat('/well/okell/users/dcs094/data/subspace/rawdata/raw_data_1-12-22/ktraj.mat')['ktraj']
    kdata = sio.loadmat('/well/okell/users/dcs094/data/subspace/rawdata/raw_data_1-12-22/kdata.mat')['kdata']
    param = sio.loadmat('/well/okell/users/dcs094/data/subspace/rawdata/raw_data_1-12-22/param.mat')['param']
    print(ktraj.shape, kdata.shape)
    print(param)
    # load coil sensitivity
    sens = mat73.loadmat('/well/okell/users/dcs094/data/subspace/rawdata/raw_data_1-12-22/meas_MID00169_FID00171_qijia_CV_VEPCASL_halfg_johnson_60_1_3_500_24x48_100hz_176_vFA_sens1.mat')['sens']
    sens = np.transpose(sens, (3,0,1,2))
    # sens: (Nc, im_size)
    print(sens.shape)

    # 1. reshape ktraj, kdata to 144 frames
    ## reformat the ktraj and kdata
    im_size = [186, 196, 150]
    # ktraj: (1398, 144, 48, 3)
    nreadouts, nt, nshots, ndim = ktraj.shape

    # kdata: (1398, 144, 48, 2, 8)
    ncoils = kdata.shape[-1]
    nframes = 144
    nsegs = 1
    # coords: (nframes,  nshots * nsegs * nreadouts , ndim)
    # inputs: (nframes,  ncoils, nshots * nsegs * nreadouts)
    coords = np.transpose(np.reshape(ktraj, (nreadouts, nframes, nsegs, nshots, ndim)), (1,3,2,0,4))
    inputs = np.transpose(np.reshape(kdata[:,:,:,1,:]-kdata[:,:,:,0,:], (nreadouts, nframes, nsegs, nshots, ncoils)), (1,4,3,2,0))
    coords = np.reshape(coords, (nframes,  nshots * nsegs * nreadouts , ndim)) * np.array(im_size)/2 / np.pi
    inputs = np.reshape(inputs, (nframes,  ncoils, nshots * nsegs * nreadouts))

    print(f"inputs shape {inputs.shape}")
    print(f"coords shape {coords.shape}")

    return inputs, coords, sens
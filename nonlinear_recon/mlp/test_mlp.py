"""
This script is used to infer the parameters of the capria angio model from the image data
"""

import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import os
from tqdm import tqdm
import nibabel as nib
from torch.utils.tensorboard import SummaryWriter

from capria_angio_to import capria_angio
from train_mlp import MLP

class param_model(nn.Module):
    def __init__(self, N) -> None:
        super(param_model, self).__init__()
        self.N = N
        self.params = nn.Parameter(torch.zeros((self.N, 3)), requires_grad=True)
        self.scaling = nn.Parameter(torch.zeros((self.N, 1)), requires_grad=True)
    
    def forward(self):
        return self.params, self.scaling

class param_infer():
    def __init__(self, img, model_path, logpath, lr=1e-3) -> None:
        # load image
        self.imshape = img.shape[:-1]
        self.Nt = img.shape[-1]
        self.img = torch.from_numpy(np.reshape(img, (-1,self.Nt)))      # [Nx,Ny,Nz, Nt]
        # define parameters to estimate
        self.N = np.prod(self.imshape)

        self.params = param_model(self.N)
        # load model
        self.model = torch.load(model_path)
        self.model.eval()
        self.model.requires_grad_(False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.optim = torch.optim.Adam(self.params.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[1500], gamma=0.1)

        self.writer = SummaryWriter(logpath)
        self.logpath = logpath

        self.loss_fn = torch.nn.MSELoss()
    def infer(self, epochs):
        self.img = self.img.to(self.device).to(torch.float32)
        self.params = self.params.to(self.device)
        for epoch in tqdm(range(epochs)):
            self.optim.zero_grad()
            params, scaling = self.params()
            out = self.model(params)
            out = out * scaling
            loss = self.loss_fn(out, self.img)
            loss.backward()
            self.optim.step()
            self.lr_scheduler.step()
            with torch.no_grad():
                self.writer.add_scalar('Loss/infer', loss.item(), epoch)
        
        self.writer.close()
        with torch.no_grad():
            p = torch.cat(self.params(), dim=1)
        return p.detach().cpu().numpy()
     
if __name__ == "__main__":
    # randomize image
    im_size = [64, 64, 64]
    delta_ts_range = [0*1e3, 2.5*1e3]
    ss_range = [1, 100]
    ps_range = [1e-3*1e3, 500e-3*1e3]
    delta_ts = np.random.uniform(delta_ts_range[0], delta_ts_range[1], (np.prod(im_size),1))
    ss = np.random.uniform(ss_range[0], ss_range[1], (np.prod(im_size),1))
    ps = np.random.uniform(ps_range[0], ps_range[1], (np.prod(im_size),1))
    scales = np.random.uniform(0.5, 100, (np.prod(im_size),1))

    angio_model = capria_angio()
    sig = angio_model.CAPRIAAngioSigAllRFAnalytic(delta_ts,ss,ps) * scales

    print("sig shape: ", sig.shape)
    infer_model = param_infer(sig, '/well/okell/users/dcs094/data/dynamic_recon/train_mlp_logs/exp2/model.pt', '/well/okell/users/dcs094/data/dynamic_recon/train_mlp_logs/infer1', lr=1)

    print("start infer ")
    p = infer_model.infer(10000)

    print("p shape: ", p.shape)
    p = np.reshape(p, im_size + [4])

    gt = np.concatenate((delta_ts, ss, ps, scales), axis=1).reshape(im_size + [4])

    nib.save(nib.Nifti1Image(p, np.eye(4)), '/well/okell/users/dcs094/data/dynamic_recon/train_mlp_logs/infer1/inferred.nii.gz')
    nib.save(nib.Nifti1Image(gt, np.eye(4)), '/well/okell/users/dcs094/data/dynamic_recon/train_mlp_logs/infer1/ground_truth.nii.gz')
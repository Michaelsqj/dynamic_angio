import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import glob
import mat73
import scipy.io as io
import imageio as io

def loadimg(fname, key=None):
    if fname.split('.')[-1] == 'mat':
        mat = mat73.loadmat(fname)
        print(mat.keys())
        if key in mat.keys():
            img = mat[key]
            print(img.shape)
            return img
        else:
            return mat
    elif fname.split('.')[-1] in ('nii','gz'):
        nii = nib.load(fname)
        img = nii.get_fdata()
        print('img shape', img.shape)
        return img

def concat_planes(mipx, mipy, mipz):
    sx = [mipx.shape[0], mipy.shape[0], mipz.shape[0]]
    sy = [mipx.shape[1], mipy.shape[1], mipz.shape[1]]
    sz_out = [max(sx), max(sy)]

    # print(sx)
    # print(sy)
    # print(sz_out)
    
    pad_sx = (((sz_out[0]-sx[0])//2, sz_out[0]-sx[0]-(sz_out[0]-sx[0])//2),
            ((sz_out[0]-sx[1])//2, sz_out[0]-sx[1]-(sz_out[0]-sx[1])//2),
            ((sz_out[0]-sx[2])//2, sz_out[0]-sx[2]-(sz_out[0]-sx[2])//2))

    pad_sy = (((sz_out[1]-sy[0])//2, sz_out[1]-sy[0]-(sz_out[1]-sy[0])//2),
              ((sz_out[1]-sy[1])//2, sz_out[1]-sy[1]-(sz_out[1]-sy[1])//2),
              ((sz_out[1]-sy[2])//2, sz_out[1]-sy[2]-(sz_out[1]-sy[2])//2))
    
    # print(pad_sx)
    # print(pad_sy)

    mipx = np.pad(mipx, pad_width=(pad_sx[0],pad_sy[0]))
    mipy = np.pad(mipy, pad_width=(pad_sx[1],pad_sy[1]))
    mipz = np.pad(mipz, pad_width=(pad_sx[2],pad_sy[2]))

    mipout = np.concatenate((mipx, mipy, mipz), axis=1)

    return mipout


def scale2uint(img, rng):
    img = (img-rng[0])/(rng[1]-rng[0])*255
    return img.astype(np.uint8)


def plot_frames(img, vmax, vmin=0, height=18, width=9.25, slice=[]):
    nframes = img.shape[-1]
    if slice == []:
        c = [img.shape[i]//2 for i in range(3)]
    else:
        c = slice
    fig = plt.figure(figsize=(height,width))
    gs = gridspec.GridSpec(3, nframes, wspace=0.0, hspace=0.0) 
    for i in range(nframes):
        ax = plt.subplot(gs[0,i])
        ax.imshow(np.transpose(np.squeeze(img[c[0],:,::-1,i])), vmin=vmin, vmax=vmax, cmap='gray')
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax = plt.subplot(gs[1,i])
        ax.imshow(np.transpose(np.squeeze(img[:,c[1],::-1,i])), vmin=vmin, vmax=vmax, cmap='gray')
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax = plt.subplot(gs[2,i])
        ax.imshow(np.transpose(np.squeeze(img[:,:,c[2],i])), vmin=vmin, vmax=vmax, cmap='gray')
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # plt.subplots_adjust(wspace=0.01,hspace=0.01)
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def mip(img):
    mipx = np.max(img, axis=0)
    mipy = np.max(img, axis=1)
    mipz = np.max(img, axis=2)
    return mipx, mipy, mipz

def plot_frames_mip(img, vmax, vmin=0, height=18, width=9.25, step=1):
    img = img[...,::step]
    nframes = img.shape[-1]
    c = [img.shape[i]//2 for i in range(3)]
    fig = plt.figure(figsize=(height,width))
    gs = gridspec.GridSpec(3, nframes, wspace=0.0, hspace=0.0) 
    mipx, mipy, mipz = mip(img)
    for i in range(nframes):
        ax = plt.subplot(gs[0,i])
        ax.imshow(np.transpose(np.squeeze(mipx[:,::-1,i])), vmin=vmin, vmax=vmax, cmap='gray')
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax = plt.subplot(gs[1,i])
        ax.imshow(np.transpose(np.squeeze(mipy[:,::-1,i])), vmin=vmin, vmax=vmax, cmap='gray')
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax = plt.subplot(gs[2,i])
        ax.imshow(np.transpose(np.squeeze(mipz[:,:,i])), vmin=vmin, vmax=vmax, cmap='gray')
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, required=True)
    parser.add_argument('--tres', type=float, required=False, default=0.5)
    parser.add_argument('--mip', type=bool, required=False, default=False)
    parser.add_argument('--nframes', type=int, required=False, default=1000)
    parser.add_argument('--outname', type=str, required=False, default="tmp.mp4")
    args = parser.parse_args()

    img = loadimg(args.fname)

    if args.mip:
        mipx, mipy, mipz = mip(img)
        mips = []
        for i in range(min(mipx.shape[-1], args.nframes)):
            mipout = concat_planes(np.transpose(mipx[:,::-1,i]),np.transpose(mipy[:,::-1,i]),np.transpose(mipz[:,:,i]))
            mipout = scale2uint(mipout, [0,np.max(mipout)])
            mips.append(mipout)
        io.mimsave(args.outname,mips[:9],duration=args.tres)

    else:
        imgs = []
        c = np.floor(np.array(img.shape)/2).astype(int)
        for i in range(min(img.shape[-1], args.nframes)):
            out = concat_planes(np.transpose(img[c[0],:,::-1,i]),np.transpose(img[:,c[1],::-1,i]),np.transpose(img[:,::-1,c[2],i]))
            out = scale2uint(out, [0,np.max(out)])
            imgs.append(out)
        # io.mimsave(args.outname,imgs, macro_block_size=1)
        io.imwrite('tmp.png',np.concatenate(imgs,axis=0))
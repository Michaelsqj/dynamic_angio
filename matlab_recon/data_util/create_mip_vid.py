import imageio
import numpy as np
import nibabel as nib
import argparse

def padimg(img, sx, sy):
    padx1, padx2 = (sx-img.shape[0])//2, sx-img.shape[0] - (sx-img.shape[0])//2
    pady1, pady2 = (sx-img.shape[0])//2, sx-img.shape[0] - (sx-img.shape[0])//2
    img = np.pad(img, pad_width=((padx1,padx2),(pady1, pady2),(0,0)))
    return img
    
def create_mip_video(img, fname):
    # create mips 
    mipx = np.transpose(np.max(img, axis=0),[1,0,2])[::-1,...]
    mipy = np.transpose(np.max(img, axis=1),[1,0,2])[::-1,...]
    mipz = np.transpose(np.max(img, axis=2),[1,0,2])[::-1,...]
    sx, sy = max(mipx.shape[0], mipy.shape[0], mipz.shape[0]), max(mipx.shape[1], mipy.shape[1], mipz.shape[1])
    mipx = padimg(mipx, sx, sy)
    mipy = padimg(mipy, sx, sy)
    mipz = padimg(mipz, sx, sy)
    
    mips = np.concatenate([mipx, mipy, mipz], axis=1)
    
    writer = imageio.get_writer(fname, fps=10)
    for i in range(mips.shape[-1]):
        img = mips[...,i]
        img = ((img-np.min(img))/(np.max(img)-np.min(img))*255).astype(np.uint8)
        writer.append_data(img)
    writer.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-i', type=str)
    p.add_argument('-o', type=str)
    args = p.parse_args()
    img = nib.load(args.i).get_fdata()
    create_mip_video(img, args.o)
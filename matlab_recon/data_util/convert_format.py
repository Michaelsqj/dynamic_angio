'''
functions to convert between .mat and .npy files
'''
import numpy as np
import scipy.io as sio
import os

def convert_mat_to_npy(mat_file, npy_file):
    '''
    convert_mat_to_npy(mat_file, npy_file)
    
    Convert .mat file to .npy file.
    
    Inputs:
        mat_file (String): Path to .mat file.
        npy_file (String): Path to .npy file.
    '''
    mat = sio.loadmat(mat_file)
    np.save(npy_file, mat)

def convert_npy_to_mat(npy_file, mat_file):
    '''
    convert_npy_to_mat(npy_file, mat_file)
    
    Convert .npy file to .mat file.
    
    Inputs:
        npy_file (String): Path to .npy file.
        mat_file (String): Path to .mat file.
    '''
    npy = np.load(npy_file)
    sio.savemat(mat_file, {"data":npy})

if __name__ == '__main__':
    npyfile = "/well/okell/users/dcs094/data/deli-cs/data/training/case000/ksp_6min.npy"
    matfile = "/well/okell/users/dcs094/data/deli-cs/data/training/case000/ksp_6min.mat"
    convert_npy_to_mat(npyfile, matfile)
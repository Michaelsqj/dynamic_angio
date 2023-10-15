# Dynamic Reconstruction of Angiogram

## Reconstruction methods

1. LLR recon, (MATLAB)

2. LLR + subspace recon (MATLAB/ sigpy)

3. LLR + MLP/Autoencoder recon (sigpy, Pytorch)

4. Extreme-MRI

## Infer method

- fabber_model_asl

## Standard datasets for test and comparison

- dataset
    - ktraj: [NCols, Nsegs*NPhases, Nshots, 3], (-pi, pi)
    - kdata: [NCols, Nsegs*NPhases, Nshots, Navgs, NCoils]
    - sens: [NCoils, Nx, Ny, Nz]

- simulation dataset
    - delta_t: mean_deltblood [0.1,1.8] s
    - p: mean_disp_p [1e-3, 500e-3]
    - s: mean_disp_s [1, 100]
    - A: mean_fblood
    
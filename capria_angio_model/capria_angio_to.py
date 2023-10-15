import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import scipy
from tqdm import tqdm

# Calculate the attenuation due to RF pulses in a CAPRIA-style acquisition.
#
# Tom Okell, June 2022
#
# R = CAPRIAAttenuation(t,t0,Alpha)
#
# where t is an array of timepoints separated by the TR, t0 is the start of
# imaging and Alpha is an array of flip angles of size(t).
class capria_angio():
    def __init__(self, TR = 14.7e-3, tau = 1.8, T1b = 1.65, N = 144, FAMode = 'Quadratic', FAParams = [3,12]) -> None:
        # Set sequence parameters
        self.TR = TR # ms
        self.tau = tau
        self.T1b = T1b
        self.N = N
        self.t0 = tau        
        self.t = np.arange(N)*TR + self.t0
        self.FAMode = FAMode
        self.FAParams = FAParams
        self.Alpha = self.CalcCAPRIAFAs(FAMode,FAParams,self.t,self.t0)

    def CAPRIAAttenuation(self,t,t0,Alpha):

        # Initialise
        R = np.zeros(shape=t.shape)
        R[0] = 1.0; 

        # Calculate attenuation due to each previous RF pulse
        for ii in range(1,len(t)):
            if t[ii] > t0:
                R[ii] = R[ii-1]*np.cos(np.deg2rad(Alpha[ii-1]))  # Attenuation
            else:
                R[ii] = 1.0

        return R

    # Returns the gamma inc function but first zeros in the elements of X which
    # are negative
    def togammainc(self,X,A):
        X[X<0] = 0.0
        A[A<0] = 0.0

        return scipy.special.gammainc(A,X)

    # This function calculates a theoretical intensities for a dynamic angio
    # voxel given the times, t, for a set of parameter values where tau is the
    # labelling duration in ms, T1 is that of blood, Alpha is the flip angle
    # (assuming spoiled GRE, in degs), TR is the repetition time, A is a scaling factor, delta_t is the
    # arrival time in ms from the labelling plane to the voxel, s is the
    # sharpness and p the time to peak of a gamma variate dispersion kernel.
    # It is assumed that all the blood sees all the RF pulses (relevant for 3D
    # acquisitions with the bottom edge of the FOV close to the labelling plane)

    def CAPRIAAngioSigAllRFAnalytic(self, delta_t,s,p):
        # delta_t: [batch_size, 1]
        # s: [batch_size, 1]
        # p: [batch_size, 1]
        # t0: scalar
        # Alpha: [Nt,1]
        # t: [Nt,1]
        
        # Define arrays for below
        delta_t = delta_t/1e3
        p = p/1e3
        a = 1+p*s
        
        # Calculate the RF attenuation term
        # R: [Nt,1]
        R = self.CAPRIAAttenuation(self.t,self.t0,self.Alpha)
    
        # Calculate the modified parameters for the integral
        # sprime: [batch_size, 1]
        sprime = s + 1.0/self.T1b
    
        # Calculate the scaling factor
        # SF: [batch_size, 1]
        SF = 2 * np.exp(-delta_t/self.T1b) * (s/sprime)**a
    
        # Calculate the incomplete gamma integrals    
        # G: [batch_size, Nt]
        G = self.togammainc(sprime*(self.t.reshape(1,-1)-delta_t),a) - self.togammainc(sprime*(self.t.reshape(1,-1)-delta_t-self.tau),a)
        #print('G:',G)
        
        # Calculate a scaling for the excitation
        E = np.sin(np.deg2rad(self.Alpha))
        
        # Output the complete result
        S = SF * R.reshape(1,-1) * G * E.reshape(1,-1)
        
        #print('S:',S.shape)
            
        return S

    # Calculate flip angle schedules for CAPRIA acquisitions
    # 
    # Tom Okell, June 2022
    #
    # Usage:
    #   Alpha = CalcCAPRIAFAs(FAMode,FAParams,t,t0)
    #
    # Required inputs:
    #   FAMode      = 'CFA', 'Quadratic' or 'Maintain'
    #   FAParams    = For CFA:          a scalar that defines the constant
    #                                   flip angle in degrees.
    #                 For Quadratic:    the flip angle varies quadratically
    #                                   between FAParams(1) and FAParams(2) in
    #                                   degrees.
    #                 For Maintain:     Uses a backwards recursive formula to
    #                                   maintain the signal at a constant level
    #                                   (i.e. magnetisation loss in the
    #                                   previous TR is counteracted by a higher
    #                                   flip angle in the next TR). In this
    #                                   case FAParams(1) defines the final flip
    #                                   angle at the end of the readout.
    #   t           = the time array to be simulated in s (assumes separation by TR)
    #   t0          = the time at which imaging commences (s)

    def CalcCAPRIAFAs(self, FAMode,FAParams,t,t0):

        # Initialise
        Alpha = np.zeros(t.shape)
        Idx = (t >=  t0)
        N = sum(Idx); # Number of pulses played out

        # CFA (FAParams = FA)
        if FAMode.upper() == 'CFA':
            Alpha[Idx] = FAParams[0] 

        # VFA quadratic (FAParams = [FAMin FAMax])
        elif FAMode.upper() == 'QUADRATIC':    
            Alpha[Idx] = FAParams[0] + (FAParams[1]-FAParams[0])*(range(N)/(N-1))**2;   

        # VFA Maintain (FAParams = FAMax)
        elif FAMode.upper() == 'MAINTAIN':
            raise Exception('Maintain not yet implemented')

        # Unknown
        else:
            raise Exception('Unknown FAMode!')


        return Alpha

if __name__ == '__main__':
    # Set sequence parameters
    TR = 14.7e-3 # ms
    tau = 1.8
    T1b = 1.65
    t0 = tau
    N = 144
    t = np.arange(N)*TR + t0
    # t = np.linspace(t0,t0+T,N)
    FAMode = 'Quadratic'
    FAParams = [3,12]

    # Physio params
    delta_ts = np.linspace(0.1,1.8,30)*1e3
    ss = np.linspace(1,100,30)
    ps = np.linspace(1e-3,500e-3,30)*1e3

    [delta_ts, ss, ps] = np.meshgrid(delta_ts,ss,ps, indexing='ij')
    angio_model = capria_angio()
    S = angio_model.CAPRIAAngioSigAllRFAnalytic(delta_ts.reshape(-1,1),ss.reshape(-1,1),ps.reshape(-1,1))
    print(S.shape)
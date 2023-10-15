"""
write the capria model as a neural network
"""
import torch
import torch.nn as nn
import numpy as np
import scipy



class DynAngioTheoreticalIntGammaAllRFAnalytic(nn.Module):
    def __init__(self, TR=14.7*1e-3, Nt=144, tau=1.8, FAParams=[3,12], FAMode='QUADRATIC') -> None:
        super().__init__()
        self.TR = TR
        self.Nt = Nt
        self.tau = tau
        self.t0 = tau
        self.t = torch.arange(Nt)*TR + self.t0
        self.T1b = 1.65

        self.FAParams = FAParams
        self.FAMode = FAMode
        self.CalcCAPRIAFAs()
        
        # Calculate the RF attenuation term
        self.CAPRIAAttenuation()

    def CalcCAPRIAFAs(self):
        # Initialise
        self.Alpha = np.zeros(self.t.shape)
        Idx = (self.t >=  self.t0)
        N = sum(Idx); # Number of pulses played out

        # CFA (FAParams = FA)
        if self.FAMode.upper() == 'CFA':
            self.Alpha[Idx] = self.FAParams[0] 

        # VFA quadratic (FAParams = [FAMin FAMax])
        elif self.FAMode.upper() == 'QUADRATIC':    
            self.Alpha[Idx] = self.FAParams[0] + (self.FAParams[1]-self.FAParams[0])*(torch.arange(N)/(N-1))**2;   

        # VFA Maintain (FAParams = FAMax)
        elif self.FAMode.upper() == 'MAINTAIN':
            raise Exception('Maintain not yet implemented')

        # Unknown
        else:
            raise Exception('Unknown FAMode!')
        # This function calculates a theoretical intensities for a dynamic angio
    # voxel given the times, t, for a set of parameter values where tau is the
    # labelling duration in ms, T1 is that of blood, Alpha is the flip angle
    # (assuming spoiled GRE, in degs), TR is the repetition time, A is a scaling factor, delta_t is the
    # arrival time in ms from the labelling plane to the voxel, s is the
    # sharpness and p the time to peak of a gamma variate dispersion kernel.
    # It is assumed that all the blood sees all the RF pulses (relevant for 3D
    # acquisitions with the bottom edge of the FOV close to the labelling plane)

    def CAPRIAAttenuation(self):

        # Initialise
        self.R = np.zeros(shape=self.t.shape)
        self.R[0] = 1.0; 

        # Calculate attenuation due to each previous RF pulse
        for ii in range(1,len(self.t)):
            if self.t[ii] > self.t0:
                self.R[ii] = self.R[ii-1]*np.cos(np.deg2rad(self.Alpha[ii-1]))  # Attenuation
            else:
                self.R[ii] = 1.0

    # Returns the gamma inc function but first zeros in the elements of X which
    # are negative
    def togammainc(X,A):
        X[X<0] = 0.0
        A[A<0] = 0.0

        return scipy.special.gammainc(A,X)


    def CAPRIAAngioSigAllRFAnalytic(self, delta_t,s,p):
        # delta_t, s, p: [batch_size,]
        # Define arrays for below
        a = 1+p*s
        
        # Calculate the modified parameters for the integral
        sprime = s + 1.0/self.T1b
    
        # Calculate the scaling factor
        SF = 2 * torch.exp(-delta_t/self.T1b) * (s/sprime)**a
    
        # Calculate the incomplete gamma integrals    
        G = self.togammainc(sprime*(self.t-delta_t),a) - self.togammainc(sprime*(self.t-delta_t-self.tau),a)
        #print('G:',G)
        
        # Calculate a scaling for the excitation
        E = torch.sin(torch.deg2rad(self.Alpha))
        
        # Output the complete result
        S = SF * self.R * G * E
        
        print('S:',S.shape)
            
        return S
    


    def forward(self, x):
        # x: [batch_size,4]
        # x[:,0:4]: delta_t, s, p, A
        # out: [batch_size, Nt]
        return x[:,3]*self.CAPRIAAngioSigAllRFAnalytic(x[:,0],x[:,1],x[:,2])

if __name__ == '__main__':
    # Test
    model = DynAngioTheoreticalIntGammaAllRFAnalytic()
    delta_ts = np.linspace(0.1,1.8,30)
    ss = np.linspace(1,100,30)
    ps = np.linspace(1e-3,500e-3,30)
    # As = np.ones_like(ps)

    [delta_ts,ss,ps] = np.meshgrid(delta_ts,ss,ps, indexing='ij')
    As = np.ones_like(delta_ts)
    x = np.stack([delta_ts.flatten(),ss.flatten(),ps.flatten(),As.flatten()],axis=1)
    y = model(torch.tensor(x,dtype=torch.float))
    print(delta_ts.shape, As.shape)
    print(y.shape)
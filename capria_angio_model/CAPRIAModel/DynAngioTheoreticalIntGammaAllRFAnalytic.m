% This function calculates a theoretical intensities for a dynamic angio
% voxel given the times, t, for a set of parameter values where tau is the
% labelling duration in ms, T1 is that of blood, Alpha is the flip angle
% (assuming spoiled GRE, in degs), TR is the repetition time, A is a scaling factor, delta_t is the
% arrival time in ms from the labelling plane to the voxel, s is the
% sharpness and p the time to peak of a gamma variate dispersion kernel.
% It is assumed that all the blood sees all the RF pulses (relevant for 3D
% acquisitions with the bottom edge of the FOV close to the labelling plane)
%
% Ihat = DynAngioTheoreticalIntGammaAllRFAnalytic(t,tau,T1b,Alpha,A,delta_t,s,p,InflowDelay)

function Ihat = DynAngioTheoreticalIntGammaAllRFAnalytic(t,tau,T1b,Alpha,A,delta_t,s,p,t0)

  Debug = false;
  if Debug; figure; end
  
  % Calculate the RF attenuation term
  R = CAPRIAAttenuation(t,t0,Alpha);
  
  % Calculate the modified parameters for the integral
  sprime = s + 1/T1b;
  
  % Calculate the scaling factor
  SF = A * exp(-delta_t/T1b) * (s/sprime)^(1+p*s);
  
  % Calculate the incomplete gamma integrals
  G = togammainc(sprime*(t-delta_t),1+p*s) - togammainc(sprime*(t-delta_t-tau),1+p*s);
  
  % Output the complete result
  Ihat = SF * R .* G;
  
  if Debug
      plot(t,[G'/max(G) R']); legend('D','R','T'); drawnow
  end
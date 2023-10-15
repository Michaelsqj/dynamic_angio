% This function calculates a theoretical intensities for a dynamic angio
% voxel given the times, t, for a set of parameter values where tau is the
% labelling duration in ms, T1 is that of blood, Alpha is the flip angle
% (assuming spoiled GRE, in degs), TR is the repetition time, A is a scaling factor, delta_t is the
% arrival time in ms from the labelling plane to the voxel, s is the
% sharpness and p the time to peak of a gamma variate dispersion kernel,
% and delta_t_min is the first arrival of blood into the imaging plane for
% this vascular component.
%
% Ihat = DynAngioTheoreticalIntGammaDeltaTMin(t,tau,T1,Alpha,TR,A,delta_t,s,p,delta_t_min)

function Ihat = DynAngioTheoreticalIntGammaDeltaTMin(t,tau,T1,Alpha,Nseg,TR,A,delta_t,s,p,delta_t_min,t0,IncludeSinAlphaTerm)

if nargin < 12; t0 = min(t); end
if nargin < 13; IncludeSinAlphaTerm = false; end

  % Convert Alpha from degrees to radians.
  Alpha = Alpha * pi / 180; % Convert degs -> rads
  
  % Calculate MaxN
  MaxN = (delta_t - delta_t_min) / TR;
  
  % Determine the number of RF pulses experienced at each time point
  P = zeros(size(t));
  P(t<=t0) = 0;
  P(t>t0) = (t(t>t0)-t0)/TR;
  N = min(P, MaxN);
  %P = (1:max(size(t))) - 1; % Index for time points
  %N = min(P * Nseg + floor(Nseg/2), MaxN); % The previous number of RF pulses before
                                % the centre of k-space is acquired in
                                % each time frame
  
  % Determine RF attenuation
  if length(Alpha) == 1 % Constant flip angle
     R = cos(Alpha).^N;
     FA = Alpha;
  else % Variable flip angle
     Idx = t > t0;
     FA = ones(size(t))*Alpha(1);
     FA(Idx) = (Alpha(1) + (Alpha(2)-Alpha(1))*((1:sum(Idx))/sum(Idx)).^2);

     % Attenuation factor due to each individual pulse
     Reach = ones(size(t));
     IdxPre = find(Idx)-1; 
     if min(IdxPre) == 0
        Idx = find(Idx);
        Ndiff = [0 N(Idx(2:end))-N(IdxPre(2:end))];
     else
        Ndiff = N(Idx)-N(IdxPre);
     end
     
     Reach(Idx) = cos(FA(Idx)).^(Ndiff);
     R = cumprod(Reach);
  end
  
  % Evaluate the function at the requested time points
  Ihat = A * ... % Scaling factor
        (togammainc(s*(t-delta_t),1+p*s) - ... % Bolus front
         togammainc(s*(t-delta_t-tau),1+p*s) ) ... % Bolus end
       * exp( -(delta_t+p) / T1  ) ... % T1 decay correction
      .* R;         % Correction for previous RF pulses

  
  % Include the sin(alpha) term if requested
  if IncludeSinAlphaTerm
      Ihat = Ihat .* sin(FA);
  end
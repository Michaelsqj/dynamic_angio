% Buxton CASL model with RF attenuation.  Note that the resulting dM
% relates to the longitudinal magnetisation and does not include the
% sin(alpha) weighting required to calculate the ASL signal

function dM = BuxtonModelWithRFAttenuationAndGammaDisp(t,f,deltat,tau,Alpha,s,p,t0,T1,T1b,lambda,T2,IncludeSinAlphaTerm)

if nargin < 8;    t0 = min(t); end
if nargin < 9  || isempty(T1);  T1 = 1.3; end
if nargin < 10 || isempty(T1b); T1b = 1.6; end
if nargin < 11 || isempty(lambda); lambda = 0.9; end
if nargin < 12 || isempty(T2); T2 = 100e-3; end
if nargin < 13 || isempty(IncludeSinAlphaTerm); IncludeSinAlphaTerm = false; end

% Calculate RF attenuation
R = CAPRIAAttenuation(t,t0,Alpha);

% Assume M0b = 1 and inversion efficiency = 1
% Calculate some useful parameters
f = f/100/60; % Rescale CBF into s^-1 units
T1prime = 1/(1/T1+f/lambda); % Apparent tissue T1
sprime = s + 1/T1b; % Modified dispersion kernel sharpness
k = 1+p*s; % Gamma kernel parameter
beta = (1 - 1/(sprime*T1prime));

% Calculate the scaling factor
SF = 2 * f * R * (s/sprime)^k * T1prime * exp(-deltat/T1b);
  
% Calculate the incomplete gamma integrals
Ga = togammainc(sprime*(t-deltat),k) - togammainc(sprime*(t-deltat-tau),k);
Gb = exp(-(t-deltat)/T1prime)/beta^k.*(togammainc(beta*sprime*(t-deltat),k) - exp(tau/T1prime)*togammainc(beta*sprime*(t-deltat-tau),k));

% Full model
dM = SF .* (Ga - Gb);

% Include the sin(alpha) term if requested
if IncludeSinAlphaTerm
    dM = dM .* sin(todeg2rad(Alpha));
end



%% Original version
% % Allow for single or variable tau
% if length(tau) < length(t); tau = ones(size(t))*tau; end
% 
% f = f/100/60;
% T1p = 1/(1/T1+f/lambda);
% 
% qss = zeros(size(t));
% Idx = (t>deltat)&(t<=tau+deltat); qss(Idx) = 1-exp(-(t(Idx)-deltat)/T1p);
% Idx = t>tau+deltat; qss(Idx) = 1-exp(-tau(Idx)/T1p);
% 
% dM = zeros(size(t));
% Idx = (t>=deltat)&(t<deltat+tau); dM(Idx) = 2*f*T1p*exp(-deltat/T1b)*qss(Idx);
% Idx = (t>=deltat+tau); dM(Idx) = 2*f*T1p*exp(-deltat/T1b)*exp(-(t(Idx)-tau(Idx)-deltat)/T1p).*qss(Idx);

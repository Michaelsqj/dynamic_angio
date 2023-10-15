% Calculate the CAPRIA signal angiographic or perfusion signal
function [dM, Alpha, R, dMTot, tAv, dMAv] = CAPRIASignal(AngioOrPerf,FAMode,FAParams,t,t0,tau,T1b,TR,params,CalcSigUsingMinAlpha,AverageOverPhases,Nsegs,Nphases,OutputAsPerc)

if nargin < 10; CalcSigUsingMinAlpha = false; end
if nargin < 11; AverageOverPhases = false; end
if nargin < 12; Nsegs = []; end
if nargin < 13; Nphases = []; end
if nargin < 14; OutputAsPerc = true; end

% Calculate flip angle scheme
Alpha = CalcCAPRIAFAs(FAMode,FAParams,t,t0);

% Calculate the signal without attenuation
if strcmpi(AngioOrPerf,'angio') % Angiographic signal
    % Calculate the angio signal with no RF attenuation for now (Alpha = 0)
    % NB. Divide by A and multiply by 2 to get into units of M0 below
    if params.delta_t_min == 0 % Use analytic solution, which includes RF attenuation
        Sig = DynAngioTheoreticalIntGammaAllRFAnalytic(t,tau,T1b,Alpha,params.A,params.delta_t,params.s,params.p,t0) / params.A * 2;        
    else        
        Sig = DynAngioTheoreticalIntGammaDeltaTMin(t,tau,T1b,0,[],TR,params.A,params.delta_t,params.s,params.p,params.delta_t_min) / params.A * 2;
    end
    
elseif strcmpi(AngioOrPerf,'perf') % Perfusion signal
    if isfield(params,'s') && isfield(params,'p') && params.delta_t_min == 0        
        Sig = BuxtonModelWithRFAttenuationAndGammaDisp(t,params.f,params.Deltat,tau,Alpha,params.s,params.p,t0,params.T1,T1b,[],[],false);
    else
        Sig = BuxtonCASLModel(t,params.f,params.Deltat,tau,params.T1,T1b);
    end
else % Unknown signal
    error('Unknown angio/perfusion type!')
end

% Calculate the attenuation factor
if (strcmpi(AngioOrPerf,'angio') && (params.delta_t_min == 0)) || (strcmpi(AngioOrPerf,'perf') && isfield(params,'s') && isfield(params,'p') && params.delta_t_min == 0)
    R = 1; % Attenuation already accounted for above
else
    R = CAPRIAAttenuation(t,t0,Alpha);
end

% Adjust Alpha for signal calculation if necessary (creates smoother signal
% profiles for visualization)
AlphaSig = Alpha;
if CalcSigUsingMinAlpha
    AlphaSig(Alpha==0) = min(Alpha(Alpha>0));
end

% Calculate the final signal strength
dM = Sig .* R .* sin(AlphaSig*pi/180);

if OutputAsPerc
    dM = dM * 100; % Units of % of M0
end

% Calculate the summed measured signal during the readout
dMTot = sum(dM(t > t0));

% Average over output phases if requested
if AverageOverPhases
    dMAv = zeros(Nphases,1);
    for ii = 1:Nphases
        Idx = (t >= t0+((ii-1)*Nsegs)*TR) & (t < t0+ii*Nsegs*TR);
        tAv(ii) = mean(t(Idx));
        dMAv(ii) = mean(dM(Idx));
    end
end
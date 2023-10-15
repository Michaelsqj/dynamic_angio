include_path()
%% set up sequenceparameters

% to_radial_matched_TR, cone
TR = 14.7e-3;  Nsegs=12; Nphases = 12; 
VFAMin = 3;     VFAMax = 12;    VFAParams = [VFAMin VFAMax];
% % to_radial_prev_prot
% TR = 10.6e-3;  Nsegs=18; Nphases = 12; 
% VFAMin = 2;     VFAMax = 9;    VFAParams = [VFAMin VFAMax];

tau = 1.8; 
t0 = tau+2e-3+10e-3+11e-3+10e-3+2e-3;  % double check with the sequence
t = t0:TR:(t0+(Nsegs*Nphases-1)*TR);
T1b = get_relaxation_times(3,'blood')/1000;     % T1 of blood at 3T

%% load 4 parameters (p,s,delta_t,A) for simulation
load_params=true;
if load_params
    load('data/todata','dt','p','s');
    delta_ts = dt(:);
    ss = s(:);
    ps = p(:);
else
    % randomize 4 parameters
end

%% simulate signal
dMs = zeros(Nphases*Nsegs, length(delta_ts));
dMAvs = zeros(Nphases, length(delta_ts));
for ii = 1:length(delta_ts)
    Aparams.p = ps(ii);
    Aparams.s = ss(ii);
    Aparams.delta_t = delta_ts(ii);
    Aparams.A = 1;
    Aparams.delta_t_min = 0; 
    [dM, VFA_Alpha, ~, ~, tAv, dMAv] = CAPRIASignal('Angio','Quadratic',VFAParams,t,t0,tau,T1b,TR,Aparams,false,true,Nsegs,Nphases,false);
    dMs(:,ii) = dM(:);
    dMAvs(:,ii) = dMAv(:);
end

%% compress data
[Nt, Ns] = size(dMAvs);

[~, S, V] = svd(dMAvs', 'econ');
disp(diag(S))
assert(size(V,1) == Nt)

ii=1;
for Nk = [2,3,4]
    B = V(:, 1:Nk);   % Nt x Nk
    err = sum( (dMAvs - B*B'*dMAvs).^2 , 'all') / sum(dMAvs(:).^2) * 100;
    disp("Nk = " + num2str(Nk) + " error = " + num2str(err)+"%");
    errs(ii)=err; ii=ii+1;
end

save('data/basis.mat','V', 'dMAvs', 'dMs');
include_path(1)

%% Set up some example parameters
tau = 1.8; 
TR = 9.1e-3; Nsegs = 36; Nphases = 6;

% Calculate the time of the start of imaging, accounting for spoilers etc. 
t0 = tau+2e-3+10e-3+11e-3+10e-3+2e-3; 

% Calculate a time array - note this must be separated by TR for the
% following model to work correctly
t = t0:TR:(t0+(Nsegs*Nphases-1)*TR);

% Flip angle array at each TR
VFAParams = [2 9];
Alpha = CalcCAPRIAFAs('quadratic',VFAParams,t,t0);

% Physiological parameters
A = 1; % Blood volume scaling parameter
delta_t = 1; % Transit time to the voxel
s = 10; % Dispersion sharpness (s^-1)
p = 100e-3; % Dispersion time to peak (s)

% T1 of blood
T1b = get_relaxation_times(3,'blood')/1000; 

% Simulate the signal
S = DynAngioTheoreticalIntGammaAllRFAnalytic(t,tau,T1b,Alpha,A,delta_t,s,p,t0).*sin(todeg2rad(Alpha));

% Alternative, more general method for calculating the CAPRIA signal for
% angio or perfusion imaging:
Aparams.A = A; Aparams.delta_t = delta_t; Aparams.s = s; Aparams.p = p; Aparams.delta_t_min = 0;
S2 = CAPRIASignal('Angio','Quadratic',VFAParams,t,t0,tau,T1b,TR,Aparams,false,true,Nsegs,Nphases,false);

% Plot - note that the scaling is slightly different for the two methods,
% so I add a factor of two here to make them the same, but it shouldn't matter, 
% as long as you are consistent
figure; plot(t,S*2,t,S2,'o');
xlabel 'time/s'; ylabel 'ASL signal/au'

%% Here is an example of simulating some data and then trying to fit it using the fabber model 
% You will need to have your own version of fabber and compile it, as per
% the main fabber instructions, including the modifications here (the
% pcasldisp branch):
% https://github.com/tomokell/fabber_models_asl/tree/pcasldisp

% Save out some trial data to fit
cd ~/Documents/Combined_Angio_And_Perfusion/Simulations/
if ~exist('Angio_fitting')
    mkdir Angio_fitting
end

cd Angio_fitting/

% Sequence parameters
tau = 1.8; 
TR = 9.1e-3; Nsegs = 36/2; Nphases = 6*2;
t0 = tau+2e-3+10e-3+11e-3+10e-3+2e-3;
t = t0:TR:(t0+(Nsegs*Nphases-1)*TR);

VFAMin = 2; VFAMax = 9; 
VFAParams = [VFAMin VFAMax];

% Physio parameters
Aparams = [];
Aparams.A = 1; Aparams.delta_ts = 0.2:0.1:1; 
Aparams.ss = 1:20; Aparams.ps = [1 10 100:200:900]*1e-3; 
Aparams.delta_t_min = 0; 
        
% T1 of blood
T1b = get_relaxation_times(3,'blood')/1000; 

clear dM VFA_Alpha tAv dMAv
for ii = 1:length(Aparams.ss)
    Aparams.s = Aparams.ss(ii);
    for jj = 1:length(Aparams.ps)
        Aparams.p = Aparams.ps(jj);
        
        
        % Angio
        for kk = 1:length(Aparams.delta_ts)
            Aparams.delta_t = Aparams.delta_ts(kk);
            [dM(:,ii,jj,kk), VFA_Alpha, ~, ~, tAv, dMAv(:,ii,jj,kk)] = CAPRIASignal('Angio','Quadratic',VFAParams,t,t0,tau,T1b,TR,Aparams,false,true,Nsegs,Nphases,false);
        end
    end
end

% Plot to check
figure;
cols = distinguishable_colors(length(Aparams.delta_ts),[1 1 1]);
ii = 2; jj = 2;
for kk = 1:length(Aparams.delta_ts)
    plot(t,squeeze(dM(:,ii,jj,kk)),'color',cols(kk,:)); hold on;
    plot(tAv,squeeze(dMAv(:,ii,jj,kk)),'o','color',cols(kk,:))
end

% Save out
save_avw(permute(dMAv,[2 3 4 1]),'Test_CAPRIA_Angio_Data','f',[1 1 1 1])
save_avw(ones(size(permute(dMAv(1,:,:,:),[2 3 4 1]))),'Test_CAPRIA_Angio_Data_Mask','b',[1 1 1 1])


%% Try running fabber manually
% Tissue only command:
fabcmd = '~/Documents/C++/fabber_models_asl/fabber_asl';
fabcmd = [fabcmd ' --data=Test_CAPRIA_Angio_Data --mask=Test_CAPRIA_Angio_Data_Mask'];
fabcmd = [fabcmd ' --model=aslrest --disp=gamma --method=vb --inferdisp']; % More freedom for dispersion
%fabcmd = [fabcmd ' --auto-init-bat']; % Try auto-init BAT
fabcmd = [fabcmd ' --batart=0.5']; % Try earlier BAT prior for arterial component
fabcmd = [fabcmd ' --noise=white --allow-bad-voxels --max-iterations=20 --convergence=trialmode --max-trials=10'];
fabcmd = [fabcmd ' --save-mean --save-mvn --save-std --save-model-fit --save-residuals'];
for jj = 1:length(tAv)
    fabcmd = [fabcmd ' --ti' ns(jj) '=' ns(tAv(jj)) ' --rpt' ns(jj) '=1']; 
end
fabcmd = [fabcmd ' --tau=' ns(tau) ' --casl --slicedt=0.0 --t1=1.3 --t1b=' ns(T1b) ' --bat=1.3 --batsd=10.0 --incbat --inferbat --incart --inferart '];
fabcmd = [fabcmd ' --capria --capriafa1=2.0 --capriafa2=9.0 --capriatr=0.0091'];
fabcmd1 = [fabcmd ' --output=fabberstep1'];

tosystem(fabcmd1)

% Convert outputs
s = ra('fabberstep1_latest/mean_disp1');
s = exp(s);
sp = ra('fabberstep1_latest/mean_disp2');
sp = exp(sp);
sp(sp > 10) = 10;
p = sp ./ s;
save_avw(s,'fabberstep1_latest/mean_disp_s','f',[1 1 1 1])
save_avw(p,'fabberstep1_latest/mean_disp_p','f',[1 1 1 1])
!fsleyes fabberstep1_latest/mean*gz fabberstep1_latest/modelfit.nii.gz Test_CAPRIA_Angio_Data.nii.gz &

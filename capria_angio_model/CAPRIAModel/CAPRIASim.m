%% Here is an example of simulating some data and then trying to fit it using the fabber model 
% You will need to have your own version of fabber and compile it, as per
% the main fabber instructions, including the modifications here (the
% pcasldisp branch):
% https://github.com/tomokell/fabber_models_asl/tree/pcasldisp

% Save out some trial data to fit
clear all
outdir = '/home/fs0/qijia/scratch/subspace/Angio_fitting50/'
if ~exist(outdir)
    mkdir(outdir)
end


% Sequence parameters
tau = 1.8; 
TR = 14.7e-3; 
Nsegs = 24/2; 
Nphases = 6*2;
t0 = tau+2e-3+10e-3+11e-3+10e-3+2e-3;
t = t0:TR:(t0+(Nsegs*Nphases-1)*TR);

VFAMin = 3; 
VFAMax = 12; 
VFAParams = [VFAMin VFAMax];

% Physio parameters
Aparams = [];
Aparams.A = 1;
num = 50;
Aparams.delta_ts = linspace(0.1,1.5,num);
Aparams.ss = linspace(1,20,num);
Aparams.ps = [1:10:100 100:50:900]*1e-3;
Aparams.delta_t_min = 0; 
% Aparams = [];
% Aparams.A = 1; Aparams.delta_ts = 0.2:0.1:1; 
% Aparams.ss = 1:20; Aparams.ps = [1 10 100:200:900]*1e-3; 
% Aparams.delta_t_min = 0; 

% T1 of blood
T1b = get_relaxation_times(3,'blood')/1000; 

clear dM VFA_Alpha tAv dMAv
for ii = 1:length(Aparams.ss)
    disp(ii);
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
saveas(gcf, fullfile(outdir, 'sim.png'));

q = matfile(fullfile(outdir, 'sim.mat'));
q.Properties.Writable = true;
q.dM = dM;
q.dMAv = dMAv;

% Save out
save_avw(permute(dMAv,[2 3 4 1]),fullfile(outdir, 'Test_CAPRIA_Angio_Data'),'f',[1 1 1 1])
save_avw(ones(size(permute(dMAv(1,:,:,:),[2 3 4 1]))),fullfile(outdir, 'Test_CAPRIA_Angio_Data_Mask'),'b',[1 1 1 1])

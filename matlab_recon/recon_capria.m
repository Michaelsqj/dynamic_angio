include_path()
%% load data
fpath = '/well/okell/users/dcs094/data/subspace/rawdata/raw_data_1-12-22/'
ind = 1;
[~, ~, p, image, kspace, ~] = loadData(ind, fpath);
%image: NCols, Nsegs*NPhases, Nshots, Navgs, NCoils
%kspace: NCols, Nsegs*NPhases, Nshots, 3
kdata = squeeze(image(:,:,:,2,:) - image(:,:,:,1,:)); %NCols, Nsegs*NPhases, Nshots, NCoils
kd = reshape(permute(kdata,[1,3,2,4]), [p.NCols*p.Nshots*12, 12, p.NCoils]);
ktraj = reshape(permute(kspace,[1,3,2,4]), [p.NCols*p.Nshots*12, 12, 3]);
recon_res = [p.res, p.res, p.res];

%%load sensitivity maps
sens_path = '/well/okell/users/dcs094/data/subspace/rawdata/raw_data_1-12-22/meas_MID00169_FID00171_qijia_CV_VEPCASL_halfg_johnson_60_1_3_500_24x48_100hz_176_vFA_sens1.mat'
load(sens_path,'sens')

%% intialise recon
tic
recon_shape = p.recon_angi_shape;
shift = p.angi_shift;
NPhases = 12;
E = xfm_NUFFT([recon_shape, NPhases], sens, [], ktraj, 'table', true, 'wi', 1);
dd = E' * kd;
t = toc; disp(['initialize took ' ns(t/60/60) ' hours']);

% rd = zeros([recon_shape,12,p.NCoils]);
% for ii = 1:p.NCoils
%     rd(:,:,:,:,ii) =  reshape(E'*(E.w.*kd(:,:,ii)), [recon_shape,12]);
% end
% rd = sum(abs(rd).^2,5).^0.5;
%% Locally Low Rank (LLR)
tic
iter = 100;
LLR.p = 5;
LLR.x = 1e-2;
rd =   pogm_LLR(E, dd, LLR.x, [1 1 1]*LLR.p, [E.Nd E.Nt], iter);
rd =   reshape(rd,[E.Nd E.Nt]);
t = toc; disp(['recon took ' ns(t/60/60) ' hours']);

save_avw(abs(rd), "/well/okell/users/dcs094/data/subspace/rawdata/raw_data_1-12-22/capria12",'d',[recon_res, 1]);

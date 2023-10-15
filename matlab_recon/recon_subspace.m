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
%% load basis
Nk = 3;
load('data/cone_12/basis.mat','V')
basis = V(:,1:Nk);

%% reconstruct
Nd = p.recon_angi_shape;
Nt = size(ktraj,2);
shift = p.angi_shift;
precompute;

%----------------------------------------------------
tic

niter = 100;
LLR.p = 5;
LLR.x = 1e-2;
rd =   pogm_LLR_sub(bpb, sens, dd, LLR.x, [1 1 1]*LLR.p, [Nd, Nk], niter);

rd =   reshape(rd,[Nd, Nk]);

t = toc; disp(['recon took ' ns(t/60/60) ' hours']);

q = matfile("/well/okell/users/dcs094/data/subspace/rawdata/raw_data_1-12-22/subspace12", 'Writable', true);
q.rd = rd;

recon_img=reshape( reshape(rd,[],Nk) * basis', [Nd,Nt] );
save_avw(abs(recon_img), "/well/okell/users/dcs094/data/subspace/rawdata/raw_data_1-12-22/subspace12",'d',[recon_res, 1]);

function capria_modelfit(fpath, prot_type, thresh, end_early)
    addpath('/home/fs0/qijia/code/CAPRIAModel')
    %%%%
    %   ｜--original image
    %   ｜--original image folder
    %   ｜----data.nii.gz
    %   ｜----mask.nii.gz
    %   ｜----fabber_out_latest
    %   ｜------mean_disp1.nii.gz
    % fpath = "/vols/Data/okell/qijia/test_fabber/meas_MID00035_FID34751_qijia_CV_VEPCASL_halfg_johnson_60_1_3_500_24x48_100hz_176_vFA_diff_LLR_avg1-0.1-5.nii.gz"  % Path to the data
    % default thresh 60 for cone
    [dirname,name,ext] = fileparts(fpath)
    name = char(name);
    name = name(1:end-length('.nii'))
    ext = ".nii.gz"
    fname = name + ext
    outpath = name+"_AngioFitting"
    mkdir(dirname+"/"+name+"_AngioFitting")
    cd(dirname+"/"+name+"_AngioFitting")

    % Rescale the data to prevent premature fitting stops
    tosystem(['fslmaths ../', char(fname), ' -mul 1e10 data'])

    %% Set up some example parameters
    tau = 1.8; 
    switch prot_type
        case 0  % cone
            TR = 14.7e-3; Nsegs = 12; Nphases = 12; VFAParams = [3 12];
        case 1  % radial
            TR = 10.6e-3;  Nsegs=18; Nphases = 12; VFAParams = [2 9];
    end

    % Calculate the time of the start of imaging, accounting for spoilers etc. 
    t0 = tau+2e-3+10e-3+11e-3+10e-3+2e-3; 

    % Calculate a time array - note this must be separated by TR for the
    % following model to work correctly
    t = t0:TR:(t0+(Nsegs*Nphases-1)*TR);
    is_subspace = false;
    if is_subspace
        tAv = t;
    else
        for ii = 1:Nphases
            Idx = (t >= t0+((ii-1)*Nsegs)*TR) & (t < t0+ii*Nsegs*TR);
            tAv(ii) = mean(t(Idx));
        end
    end

    Alpha = CalcCAPRIAFAs('quadratic',VFAParams,t,t0);

    % T1 of blood
    T1b = get_relaxation_times(3,'blood')/1000; 

    % Generate a mask
    % !fslmaths CAPRIA_Recon/183_FA2_9_Angio/sens_mag.nii.gz -Tmax -bin mask
    tosystem(['fslmaths data -Tmax -thr ',ns(thresh),' -bin mask'])
    % !fslmaths data -Tmax -thr 30 -bin mask
    !cluster --in=mask --thresh=0.1 -o mask_clusters
    [~,tmp]=builtin('system','fslstats mask_clusters -R');
    maxI = split(tmp,' ');
    maxI = str2num(maxI{2});
    thr = ns(maxI - 1);
    thr = ns(2);
    tosystem(['fslmaths mask_clusters -thr ' thr ' -bin mask_clusters_bin'])
    !fslmaths data -Tmax -thr 50 -bin -ero -dilF maskerodil

    if end_early
        return
    end
    %% Run the fitting
    fabcmd = '/home/fs0/qijia/code/fsldev/bin/fabber_asl';
    fabcmd = [fabcmd ' --data=data.nii.gz --mask=mask_clusters_bin.nii.gz'];
    fabcmd = [fabcmd ' --model=aslrest --disp=gamma --method=vb --inferdisp']; 
    fabcmd = [fabcmd ' --batart=0.5']; % Try earlier BAT prior for arterial component
    fabcmd = [fabcmd ' --noise=white --allow-bad-voxels --max-iterations=20 --convergence=trialmode --max-trials=10'];
    fabcmd = [fabcmd ' --save-mean --save-mvn --save-std --save-model-fit --save-residuals'];
    for jj = 1:length(tAv)
        fabcmd = [fabcmd ' --ti' ns(jj) '=' ns(tAv(jj)) ' --rpt' ns(jj) '=1']; 
    end
    fabcmd = [fabcmd ' --tau=' ns(tau) ' --casl --slicedt=0.0 --t1=1.3 --t1b=' ns(T1b) ' --bat=1.3 --batsd=10.0 --incbat --inferbat --incart --inferart '];
    fabcmd = [fabcmd ' --capria --capriafa1=' ns(VFAParams(1)) ' --capriafa2=' ns(VFAParams(2)) ' --capriatr=' ns(TR)];
    fabcmd1 = [fabcmd ' --output=fabber_out'];
    % disp(fabcmd1)
    tosystem(fabcmd1)

    % Convert outputs
    s = ra('fabber_out_latest/mean_disp1');
    s = exp(s);
    sp = ra('fabber_out_latest/mean_disp2');
    sp = exp(sp);
    sp(sp > 10) = 10;
    p = sp ./ s;
    save_avw(s,'fabber_out_latest/mean_disp_s','f',[1 1 1 1])
    save_avw(p,'fabber_out_latest/mean_disp_p','f',[1 1 1 1])
    !fslcpgeom fabber_out_latest/mean_disp2 fabber_out_latest/mean_disp_s -d
    !fslcpgeom fabber_out_latest/mean_disp2 fabber_out_latest/mean_disp_p -d
end
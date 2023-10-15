function saveData(ind, fpath, outpath)
    %saveData(ind, fpath)
    %   ind: index of the data to be saved
    %   fpath: path to the data
    %   load data from scanner raw data, save the kdata, ktraj in .mat format
    include_path();
    [~, ~, p, image, kspace, ~] = loadData(ind, fpath);
    %image: NCols, Nsegs*NPhases, Nshots, Navgs, NCoils
    %kspace: NCols, Nsegs*NPhases, Nshots, 3
    % kdata = squeeze(image(:,:,:,2,:) - image(:,:,:,1,:)); %NCols, Nsegs*NPhases, Nshots, NCoils
    % kd = reshape(permute(kdata,[1,3,2,4]), [p.NCols*p.Nshots*12, 12, p.NCoils]);
    % ktraj = reshape(permute(kspace,[1,3,2,4]), [p.NCols*p.Nshots*12, 12, 3]);
    kdata = image;
    ktraj = kspace;
    param = p;
    save([char(outpath),'/','kdata'], 'kdata');
    save([char(outpath),'/','ktraj'], 'ktraj'); % [-pi, pi]
    save([char(outpath),'/','param'], 'param');
end
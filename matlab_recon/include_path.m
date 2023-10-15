function include_path(inv)
    paths = ["/well/okell/users/dcs094/irt/nufft",...
    "/well/okell/users/dcs094/irt/utilities",...
    "/well/okell/users/dcs094/irt/systems",...
    genpath('/well/okell/users/dcs094/irt/mex'),...
    genpath(".")];
    for p = paths
        if nargin<1 || inv==0
            addpath(p)
        else
            rmpath(p)
        end
    end
end
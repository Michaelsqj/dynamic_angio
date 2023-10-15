function [base_k] = convert_g2k(base_g)
    %------------------------------
    %   convert gradient to kspace
    %       gradient: G/cm
    %       kspace: 1/cm
    %
    %------------------------------
    gamma = 42.58*1e2; % Hz/G
    T  = 10e-3; % gradient rapser time, ms
    base_k(:,1) = cumtrapz(squeeze(base_g(:,1))) .* 4.258 .* T;
    base_k(:,2) = cumtrapz(squeeze(base_g(:,2))) .* 4.258 .* T;
    base_k(:,3) = cumtrapz(squeeze(base_g(:,3))) .* 4.258 .* T;
    
end
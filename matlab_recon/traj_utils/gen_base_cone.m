function [time, g, k] = gen_base_cone(p)
    %-------------------------------------------------------------
    % Code to make a base cone
    %
    % Inputs:
    %   mat = resolution
    %   fov    = field of view in [mm]
    %   smax   = G/cm/ms
    %   gmax =   G/cm
    %   readout_time = time for readout in ms
    %   cone_angle = angle at edge of k-space in degrees
    %   Ts = sampling time [ms]
    %
    %Outputs:
    %   k = 3d k space trajectory
    %   time = time points in s
    %   g = required gradients
    
    if contains(p.cone_type,'radial')
        [time, g, k] = gen_radial(p.mat, p.fov, p.smax, p.gmax, floor(p.readout_time/p.Ts), p.Ts);
    else
        [time, g, k] = feval(p.cone_type, p.mat, p.fov, p.Ts, p.gmax, p.smax, p.readout_time, p.cone_angle);
    end
end
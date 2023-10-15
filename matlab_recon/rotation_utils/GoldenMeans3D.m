% Returns azimuthal and polar angles for N 3D golden means radial spokes as per Chan et
% al. MRM 2009.  If N is an array, this is taken as the golden ratio index
% array.

% function [Azi, Polar] = GoldenMeans3D(N,ReverseOddLines)

% if nargin < 2; ReverseOddLines = false; end

% % Define the increments, from Chan et al
% Phis = GoldenRatios3D;

% % Calculate Polar and Azimuthal angles
% % NB. apparent error in Chan paper figure here - Beta is the angle from the
% % kz axis = Polar angle in Siemens terms
% if length(N) == 1
%     m = (0:(N-1))';
% else
%     m = N(:);
% end

% kz = mod(m*Phis(1),1); % Use GR to evenly distribute between 0 and 1
% % Can potentially invert the sign and add pi to every other azimuthal angle
% % to get samples in both directions if desired, but leave this until
% % later...

% Polar = acos(kz); 
% Azi   = mod(m*Phis(2),1) * 2 * pi;

% % Reverse every other line if requested
% if ReverseOddLines
%     OddIdx = logical(mod(m,2));
    
%     % Add pi to the azimuthal angle
%     Azi(OddIdx) = mod( Azi(OddIdx) + pi, 2*pi);
    
%     % Reverse kz
%     Polar(OddIdx) = acos(-kz(OddIdx)); 
% end
function [Azi, Polar] = GoldenMeans3D(N, ttype)
    % type 1: distribution on half sphere
    % type 2: distribution on whole sphere
    % Define the increments, from Chan et al
    Phis = GoldenRatios3D;

    % Calculate Polar and Azimuthal angles
    % NB. apparent error in Chan paper figure here - Beta is the angle from the
    % kz axis = Polar angle in Siemens terms

    m = N(:);
    if ttype == 1
        kz = mod(m*Phis(1),1);
    else
        kz = mod(m*Phis(1),1) * 2 - 1;
    end
    
    Polar = acos(kz); 
    Azi   = mod(m*Phis(2),1) * 2 * pi;

    if ttype == 1
        % Reverse every other line if requested
        OddIdx = logical(mod(m,2));

        % Add pi to the azimuthal angle
        Azi(OddIdx) = mod( Azi(OddIdx) + pi, 2*pi);

        % Reverse kz
        Polar(OddIdx) = acos(-kz(OddIdx)); 
        
    end

end
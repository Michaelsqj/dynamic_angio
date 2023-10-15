function [T1 T2] = get_relaxation_times(field,tissue)

%  function [T1 T2] = get_relaxation_times(field,tissue)
%
%  field can be 1.5, 3 or 7; tissue can be 'wm', 'gm', or 'blood'

% now specify defaults before checking for other params
switch(field)
 case {1.5}
   T1gm = 900;    T2gm = 110;  % T1 from breger89; T2 scaled from wansapura99
   T1wm = 650;    T2wm = 90;   % T2 from breger89; T2 scaled from wansapura99
   T1b  = 1300;   T2b  = 250;
 case {3}
   T1gm = 1330;   T2gm =  90;  % from wansapura99
   T1wm = 830;    T2wm =  80;  % from wansapura99
   T1b  = 1650;   T2b  = 150;  % T.O. as per asl_calib
   T1csf = 4300;  T2csf = 750; % T.O. as per asl_calib
 case {7}
   T1gm = 1650;    T2gm = 60;  % T2 from pfeuffer04, T1 from Wright06
   T1wm = 1020;    T2wm = 55;  % T2 from pfeuffer04, T1 from Wright06
   T1b  = 2200;    
   T2b  = 80; % R20blood (at 7T, Hct = 0.45, fully oxygenated) = 12.49 s^-1, from Grgac, K et al. Magn. Reson. Imaging 38, 234?249 (2017).
  otherwise
   error('ERR_FLD','unknown field strength specified');
end;

switch(lower(tissue))
 case{'gm'}
   T1=T1gm; T2=T2gm;
 case{'wm'}
   T1=T1wm; T2=T2wm;
 case{'blood'}
   T1=T1b; T2=T2b;
 case{'csf'} % T.O.
   T1=T1csf; T2=T2csf;     
 otherwise
   error('ERR_TISS','unknown tissue type specified');
end;


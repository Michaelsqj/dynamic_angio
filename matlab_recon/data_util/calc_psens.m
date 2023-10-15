function [psens sens_xfm s] = calc_psens(sens)

dims            =   size(sens);
[u,s,sens_xfm]  =   lsvd(reshape(sens, [], dims(end)));
psens           =   reshape(u*s, dims);
% psens           =   u*s;
s               =   cumsum(abs(diag(s)).^2)/sum(abs(diag(s)).^2);

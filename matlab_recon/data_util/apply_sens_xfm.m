function dd = apply_sens_xfm(xfm, dd, p, coil_dim)

dims    =   size(dd);
dims(3) =   size(dd,3);
dims(4) =   size(dd,4);
dims(coil_dim) = p;

x       =   [setdiff(1:4, coil_dim) coil_dim];
y(x)    =   1:4;
dd      =   permute(reshape(reshape(permute(dd,x),[],size(dd,coil_dim))*xfm(:,1:p),dims(x)),y);

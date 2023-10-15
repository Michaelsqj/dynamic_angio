function step = max_step(iters, bpb, C, msize)
    % msize   =   [prod(res.Nd) res.Nt];
    %   Use the power method to find the max eigenvalue of E'E
    y   =   randn(msize);
    N   =   0;
    ii  =   0;
    while abs(norm(y(:)) - N)/N > 1E-4 && ii < iters
        N   =   norm(y(:)); 
        % y   =   xfm.mtimes2(y/N);
        y   =   mtimes2(y/N, bpb, C);
        ii  =   ii+1;
    end
    step    =   1./norm(y(:));
end
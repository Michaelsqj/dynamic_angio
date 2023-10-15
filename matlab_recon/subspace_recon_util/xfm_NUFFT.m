classdef xfm_NUFFT < xfm
%   NUFFT Linear Operator
%   Forward transforms images to multi-coil, arbitrary non-Cartesian k-space
%   along with the adjoint (multi-coil k-space to combined image)
%
%   Mark Chiew
%   mchiew@fmrib.ox.ac.uk
%   July 2015
%
%   NB: Requires the nufft portion of Jeff Fessler's irt toolbox
%       See http://web.eecs.umich.edu/~fessler/irt/fessler.tgz
%
%   Required Inputs:
%       dims    =   [Nx, Ny, Nz, Nt] 4D vector of image dimensions
%       k       =   [Nsamp, Nt, 2] (2D) or [Nsamp, Nt, 3] (3D) 
%                   array of sampled k-space positions
%                   (in radians, normalised range -pi to pi)
%
%   Optional Inputs:
%       coils   =   [Nx, Ny, Nz, Nc] array of coil sensitivities 
%                   Defaults to single channel i.e. ones(Nx,Ny,Nz)
%       wi      =   Density compensation weights
%       Jd      =   Size of interpolation kernel
%                   Defaults to [6,6]
%       Kd      =   Size of upsampled grid
%                   Defaults to 200% of image size
%       shift   =   NUFFT shift factor
%                   Defaults to 50% of image size
%
%   Usage:
%           Forward transforms can be applied using the "*" or ".*" operators
%           or equivalently the "mtimes" or "times" functions
%           The input can be either the "Casorati" matrix (Nx*Ny*Nz, Nt) or
%           an n-d image array
%           The adjoint transform can be accessed by transposing the transform
%           object 
%           The difference between "*" and ".*" is only significant for the 
%           adjoint transform, and changes  the shape of the image ouptut:
%               "*" produces data that is in matrix form (Nx*Ny*Nz, Nt) whereas
%               ".*" produces data in n-d  array form (Nx, Ny, Nz, Nt) 

properties (SetAccess = protected, GetAccess = public)
    k       =   [];
    w       =   [];
    norm    =   1;
    Jd      =   [6,6];
    Kd      =   [];
    shift   =   [];
    PSF     =   []; % Eigenvalues of circulant embedding 
    tbl     =   [];
    loop    =   [];
    st      =   [];
end

methods
function res = xfm_NUFFT(dims, coils, fieldmap_struct, k, varargin)

    %   Base class constructor
    res =   res@xfm(dims, coils, fieldmap_struct);
    
    %   Parse remaining inputs
    p   =   inputParser;

    %   Input validation functions
    lengthValidator =   @(x) length(x) == 2 || length(x) == 3;

    %   Input options
    p.addParamValue('wi',       [],                     @(x) size(x,2) == dims(4)||isscalar(x));
    p.addParamValue('Jd',       [6,6,6],                lengthValidator);
    p.addParamValue('Kd',       2*dims(1:3),            lengthValidator);
    p.addParamValue('shift',    floor(dims(1:3)/2),     lengthValidator);
    p.addParamValue('mean',     true,                   @islogical);
    p.addParamValue('table',    false,                  @islogical);
    p.addParamValue('loop',     false,                  @islogical);
    p.addParamValue('PSF',      []);
    p.addParamValue('st',       []);

    p.parse(varargin{:});
    p   =   p.Results;

    res.Jd      =   p.Jd;
    res.Kd      =   p.Kd;
    res.shift   =   p.shift;

    res.k       =   k;
    res.dsize   =   [size(k,1) res.Nt res.Nc];

    res.tbl     =   p.table;
    res.loop    =   p.loop;

    res.PSF     =   p.PSF;
    res.st      =   p.st;

    if isempty(res.st)
        disp('Initialising NUFFT(s)');
        nd  =   (res.Nd(3) > 1) + 2;
        for t = res.Nt:-1:1
            if ~res.tbl
                st(t)   =   nufft_init(squeeze(k(:, t, 1:nd)),...
                                       res.Nd(1:nd),...
                                       p.Jd(1:nd),...
                                       p.Kd(1:nd),...
                                       p.shift(1:nd));
            else
                st(t)   =   nufft_init(squeeze(k(:, t, 1:nd)),...
                                       res.Nd(1:nd),...
                                       p.Jd(1:nd),...
                                       p.Kd(1:nd),...
                                       p.shift(1:nd),...
                                       'table',2^11,'minmax:kb');
            end
        end
        res.st  =   st;
    end
    if isempty(p.wi)
    disp('Generating Density Compensation Weights');
    %   Use (Pipe 1999) fixed point method
        for t = 1:res.Nt
            res.w(:,t)  =   ones(size(k,1),1);
            for ii = 1:10
                tic
                if ~res.tbl
                    res.w(:,t)  =   res.w(:,t)./real(res.st(t).p*(res.st(t).p'*res.w(:,t)));
                else
                    res.w(:,t)  =   res.w(:,t)./real(res.st(t).interp_table(res.st(t),res.st(t).interp_table_adj(res.st(t),res.w(:,t))));
                end
                ts=toc; disp(["iter "+num2str(ii)+" took "+num2str(ts/60)+" min"]);
            end
        end
        res.w   =   res.w*res.st(1).sn(ceil(end/2),ceil(end/2),ceil(end/2))^(-2)/prod(res.st(1).Kd);
    elseif p.wi == 0
        w   =   ones(size(k,1),1);
        for ii = 1:20
            tic
            w  =   w./real(res.st(t).p*(res.st(t).p'*w));
            ts=toc; disp(["iter "+num2str(ii)+" took "+num2str(ts/60)+" min"]);
        end
        res.w   =   res.w*res.st(1).sn(ceil(end/2),ceil(end/2),ceil(end/2))^(-2)/prod(res.st(1).Kd);
    elseif isscalar(p.wi)
        res.w   =   repmat(p.wi, 1, res.Nt);
    else
        res.w   =   reshape(p.wi, [], res.Nt);
    end
    res.w       =   sqrt(res.w);

    if isempty(res.PSF)
        t0 = tic();
        res.PSF =   res.calcToeplitzEmbedding();
        dt = toc(t0); disp(["calcToeplitzEmbedding took "+num2str(dt/60)+" min"]);
    end
end

function T = calcToeplitzEmbedding(a)
    disp('Computing Toeplitz Embedding')
    Nd  =   a.Nd;
    Nt  =   a.Nt;
    st  =   a.st;
    w   =   a.w.^2;

    %   Need 2^(d-1) columns of A'A
    %   4 columns for 3D problems
    x1  =   zeros([Nd(1) prod(Nd(2:3)) Nt],'single');
    x2  =   zeros([Nd(1) prod(Nd(2:3)) Nt],'single');
    x3  =   zeros([Nd(1) prod(Nd(2:3)) Nt],'single');
    x4  =   zeros([Nd(1) prod(Nd(2:3)) Nt],'single');

    T   =   zeros(8*prod(Nd), Nt,'single');

    %   First column
    tic
    tmp =   zeros(Nd,'single');
    tmp(1,1,1)  =   1;
    for t = 1:Nt
        x1(:,:,t)   =   reshape(nufft_adj(w(:,t).*nufft(tmp, st(t)), st(t)), Nd(1), []);
    end
    ts=toc; disp(["First column took "+num2str(ts/60)+" min"]);

    %   Second column
    tic
    tmp =   zeros(Nd,'single');
    tmp(end,1,1)    =   1;
    for t = 1:Nt
        x2(:,:,t)   =   reshape(nufft_adj(w(:,t).*nufft(tmp, st(t)), st(t)), Nd(1), []);
        x2(end,:,t) =   0;
    end
    ts=toc; disp(["Second column took "+num2str(ts/60)+" min"]);

    %   Third column
    tic
    tmp =   zeros(Nd,'single');
    tmp(1,end,1)    =   1;
    for t = 1:Nt
        x3(:,:,t)   =   reshape(nufft_adj(w(:,t).*nufft(tmp, st(t)), st(t)), Nd(1), []);
    end
    ts=toc; disp(["Third column took "+num2str(ts/60)+" min"]);

    %   Fourth column
    tic
    tmp =   zeros(Nd,'single');
    tmp(end,end,1)  =   1;
    for t = 1:Nt
        x4(:,:,t)   =   reshape(nufft_adj(w(:,t).*nufft(tmp, st(t)), st(t)), Nd(1), []);
        x4(end,:,t) =   0;
    end
    ts=toc; disp(["Fourth column took "+num2str(ts/60)+" min"]);

    %   Perform first level embedding
    M1  =   cat(1, x1, circshift(x2,1,1));
    clear x1 x2;
    M2  =   cat(1, x3, circshift(x4,1,1));
    clear x3 x4;


    %   Perform second level embedding
    M2  =   reshape(M2, [2*Nd(1) Nd(2:3) Nt]);
    M2(:,end,:,:)   =   0;
    M1  =   reshape(M1, [], Nd(3), Nt);
    M2  =   reshape(M2, [], Nd(3), Nt);
    M3  =   cat(1, M1,  circshift(M2,2*Nd(1),1));
    
    clear M1 M2;

    %   Perform third (final) level embedding
    M3  =   reshape(M3, 2*Nd(1), 2*Nd(2), Nd(3), Nt);

    T(1:4*prod(Nd),:) = reshape(M3, [], Nt);

    M3  =   circshift(flipdim(M3,3),1,3);
    M3  =   circshift(flipdim(M3,2),1,2);
    M3  =   circshift(flipdim(M3,1),1,1);

    for i = 1
        T(4*prod(Nd)+4*(i-1)*prod(Nd(1:2))+1:4*prod(Nd)+4*i*prod(Nd(1:2)),:)    =   0;
    end
    for i = 2:Nd(3)
        T(4*prod(Nd)+4*(i-1)*prod(Nd(1:2))+1:4*prod(Nd)+4*i*prod(Nd(1:2)),:)    =   conj(reshape(M3(:,:,i,:),[],Nt));
    end

    T   =   prod(sqrt(2*Nd))*a.fftfn_ns(reshape(T,[2*Nd Nt]), 1:3)*a.norm^2;

end


function b = mtimes2(a,b)
    %   If mtimes(A,b) = A*b, mtimes2(A,b) = A'A*b
    %   If Toeplitz embedding is available, uses that
    %   otherwise computes by mtimes(A',mtimes(A,b))
    PSF =   a.PSF;
    Nt  =   a.Nt;
    Nd  =   a.Nd;
    Nc  =   a.Nc;
    S   =   a.S;
    dim =   size(b);
    b   =   reshape(b,[],Nt);
    
    if Nd(3) == 1
    %   2D mode
        tmp =   zeros(2*Nd(1),2*Nd(2),1,1,Nc);
        tmp2=   zeros(2*Nd(1),2*Nd(2),1,1,Nc);
        for t = 1:Nt
            tmp(1:Nd(1),1:Nd(2),1,1,:)  =  S*b(:,t); 
            tmp2(:,:,1,1,:) =   ifft2(PSF(:,:,1,t).*fft2(tmp)); 
            b(:,t)  =   reshape(S'*tmp2(1:Nd(1),1:Nd(2),1,1,:),[],1);
        end
    else
    %   3D mode, break out coil loop for reduced memory footprint
        for t = 1:Nt
            out =   zeros(size(b,1),1);
            for c = 1:Nc
                tmp =   zeros(2*Nd(1),2*Nd(2),2*Nd(3));
                tmp(1:Nd(1),1:Nd(2),1:Nd(3))  =  mtimes(S,b(:,t),c); 
                tmp =   ifftn(PSF(:,:,:,t).*fftn(tmp)); 
                out =   out + reshape(mtimes(S',tmp(1:Nd(1),1:Nd(2),1:Nd(3)),c),[],1);
            end
            b(:,t)  =   out;
        end
    end

    %   Return b in the same shape it came in
    b   =   reshape(b, dim);    
end

function res = mtimes(a,b,idx)
    if nargin < 3
        idx =   1:a.Nt;
    end
    %   Property access in MATLAB classes is very slow, so any property accessed
    %   more than once is copied into the local scope first
    nt  =   length(idx);
    st  =   a.st(idx);

    if a.adjoint
    %   Adjoint NUFFT and coil transform
        res =   zeros([a.Nd, nt a.Nc]);
        b   =   bsxfun(@times, b, a.w(:,idx));
        for t = 1:nt
            t0 = tic();
            if a.tbl || a.loop
                for c = 1:a.Nc
                    res(:,:,:,t,c)  =   nufft_adj(squeeze(b(:,t,c)), st(t));
                end
            else
                res(:,:,:,t,:)  =   nufft_adj(squeeze(b(:,t,:)), st(t));
            end
            dt = toc(t0); disp(["Adjoint NUFFT t=" num2str(t) ' took ' num2str(dt/60) " min"]);
        end
        res =   reshape(a.norm*(a.S'*res), [], nt);
    else
    %   Forward NUFFT and coil transform
        res =   zeros([a.dsize(1) nt a.dsize(3)]);
        tmp =   a.norm*(a.S*b);
        for t = 1:nt
            if a.tbl || a.loop
                for c = 1:a.Nc
                    res(:,t,c)  =   nufft(squeeze(tmp(:,:,:,t,c)), st(t));
                end
            else
                res(:,t,:)  =   nufft(squeeze(tmp(:,:,:,t,:)), st(t));
            end
        end
        res =   bsxfun(@times, res, a.w(:,idx));
    end

end

function res = mean(a,b)
    nd  =   (a.Nd(3) > 1) + 2;
    st  =   nufft_init(reshape(a.k,[],nd),...
                       a.Nd(1:nd),...
                       a.Jd(1:nd),...
                       a.Kd(1:nd),...
                       a.shift(1:nd));
    %   Use (Pipe 1999) fixed point method
    w   =   ones(numel(a.k)/nd,1);
    for ii = 1:20
        tmp =   st.p*(st.p'*w);
        w   =   w./real(tmp);
    end
    w   =   w*sqrt(st.sn(ceil(end/2),ceil(end/2))^(-2)/prod(st.Kd));
    res =   a.S'*(nufft_adj(bsxfun(@times, reshape(b,[],a.Nc), w), st));
end

end
end

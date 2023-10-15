function [x] = mtimes2(alpha, bpb, C)
    % alpha: [Nx, Ny, Nz, Nk], coeffcient maps
    % bpb: [2*Nx, 2*Ny, 2*Nz, Nk, Nk], PSF
    % C: {Nx, Ny, Nz, Nc}, Coil sensitivity maps
    t0 = tic();
    Nd = size(C, 1:3);
    Ndprod = prod(Nd);
    Ndk = size(bpb, 1:3);
    Nk = size(bpb, 4);
    Nc = size(C, 4);
    
    C = reshape(C, [Nd, 1, Nc]);
    alpha = reshape(alpha, [Nd, Nk]);

    x = C.*alpha;   % [Nx, Ny, Nz, Nk, Nc]
    x = reshape(x, [Nd, Nk*Nc]);

    disp("fftn x")
    xk = zeros([Ndk, Nk*Nc]);
    xk(1:Nd(1),1:Nd(2),1:Nd(3),:) = x;
    xk = batch_fftn(xk);
    xk = reshape(xk, [Ndk, 1, Nk, Nc]); % [Nx, Ny, Nz, 1, Nk, Nc]

    disp("multiply bpb with xk")
    % multiply bpb with x
    % bpb [Nd, Nk, Nk]
    % if Nk <=4
        xk = squeeze(sum(bpb .* xk, 5));   % [Nx, Ny, Nz, Nk, Nc]
    % else
    %     tmp = zeros([Ndk, Nk, 1, Nc]);
    %     for ii =1:Nk
    %         tmp = tmp + bpb(:,:,:,:,ii) .* xk(:,:,:,:,ii,:);
    %     end
    %     xk = squeeze(tmp);
    %     clear tmp
    % end
    xk = reshape(xk, [Ndk, Nk*Nc]);
    % xk = permute(xk, [5,4,1,2,3,6]); % [Nk, 1, [Ndk], Nc]
    % bpb = permute(bpb, [4,5,1,2,3]); % [Nk, Nk, [Ndk]]
    % xk = pagemtimes(bpb, xk); % [Nk, 1, [Ndk], Nc]
    % xk = reshape((permute(xk, [3,4,5,1,2,6])), [Ndk, Nk*Nc]);

    disp("ifftn")
    xk = reshape(batch_ifftn(xk), [Ndk, Nk, Nc]); % [Nx, Ny, Nz, Nk, Nc]
    x = xk(1:Nd(1),1:Nd(2),1:Nd(3),:,:);

    x = sum( conj(C).*x, 5 );   % [Nx, Ny, Nz, Nk]
    % x = reshape(x, [Ndprod, Nk]);
    dt = toc(t0); disp(['mtimes2 took ' num2str(dt/60) 'min'])
end

function [tmp] = batch_fftn(x)
    Nb = size(x, 4);
    Nd = size(x, 1:3);
    tmp = zeros([Nd, Nb]);
    for ii = 1:Nb
        tmp(:,:,:,ii) = fftn_fast(x(:,:,:,ii));
    end
end

function [tmp] = batch_ifftn(x)
    Nb = size(x, 4);
    Nd = size(x, 1:3);
    tmp = zeros([Nd, Nb]);
    for ii = 1:Nb
        tmp(:,:,:,ii) = ifftn_fast(x(:,:,:,ii));
    end
end
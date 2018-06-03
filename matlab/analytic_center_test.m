clear all; close all;

n = 4;
M = 1;          %BPSK
k = 6;         %10 symbols
t = 1;
num_steps = 3;

%Random square channel
H = randn(n);

%Make symbols
X = 2*randi(M+1,n,k) - 3;
M = M + 1;

%Test with no noise 
Y = H*X;

% Starting value
Astart = inv(randn(n));
startCount = 0;
while any(max(abs(Astart*Y)) > M) || rank(Astart) < n
    Astart = inv(randn(n));
    startCount = startCount + 1;
end
    
%%
Ak = Astart;
num_steps = 10;
res = NaN(num_steps+1,1);
res(1) = log(sum(M - max(abs(Ak*Y))));
ndec = NaN(num_steps,1);
for iter = 1:num_steps
    % Compute gradient, size: n^2

    grad_phi = zeros(n,n);
    for j = 1:n
        aj = Ak(:,j);
        for i = 1:k
            yi = Y(:,i);
            grad_phi(j,:) = grad_phi(j,:) + (-yi/(M - aj'*yi) + yi/(M + aj'*yi))';
        end
    end
    grad_phi = reshape(grad_phi, [n*n,1]);

    % Compute Hessians
    Hphi = zeros(n*n);
    for j = 1:n
        aj = Ak(:,j);
        for i = 1:k
            yi = Y(:,i);            
            HRow = -yi*yi'/(M - aj'*yi)^2 - yi*yi'/(M + aj'*yi)^2;

            for ii = 0:n-1
            for jj = 0:n-1
                row = j+ii*4; col = j+jj*4;
                Hphi(row,col) = Hphi(row,col) + HRow(ii+1,jj+1);
            end
            end
        end
    end

    Hf = -Hphi;
    gf = -grad_phi;
    
%     % Compute Newton Steps
    Hf_inv = inv(Hf);
    Unt = -Hf_inv*gf;
    ndec(iter) = sqrt(gf' * Hf_inv * gf);
    tau = 0.1;
%     while (-log(abs(det(Uk))) < -log(abs(det(Uk - tau * reshape(Unt, [n,n])))))
%         tau = 0.9 * tau;
%     end
    Ak = Ak - tau * reshape(Unt, [n,n]);
%     
%     Uk = Uk - 0000.1 * reshape(gf, [n,n]);
    res(iter + 1) = -log(sum(M - max(Ak*Y))) - log(sum(M + min(Ak*Y)));
end
res
Ak


%%

cvx_begin
    variable U(n,n) 
%     minimize(-log_det(U))
    subject to
        norm(U*Y, Inf) <= 1
cvx_end
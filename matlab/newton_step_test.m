clear all; close all;

n = 4;
M = 1;          %BPSK
k = 10;         %10 symbols
t = 500;
num_steps = 10;

%Random square channel
H = randn(n);

%Make symbols
X = 2*randi(M+1,n,k) - 3;
M = M + 1;

%Test with no noise 
Y = H*X;

% Starting value
Uk = inv(randn(n));
startCount = 0;
while any(max(abs(Uk*Y)) > M) || rank(Uk) < n
    Uk = inv(randn(n));
    startCount = startCount + 1;
end
    

res = NaN(num_steps,1);
ndec = NaN(num_steps,1);
for iter = 1:num_steps
    % Compute gradient, size: n^2
    Uk_inv = inv(Uk);
    grad_f0 = reshape(-Uk_inv', [n*n,1]);

    grad_phi = zeros(n,n);
    for j = 1:n
        uj = Uk(j,:);
        for i = 1:k
            yi = Y(:,i);
            grad_phi(j,:) = grad_phi(j,:) + (-yi/(M - uj*yi) + yi/(M + uj*yi))';
        end
    end
    grad_phi = reshape(grad_phi, [n*n,1]);

    gf = t*grad_f0 - grad_phi;

    % Compute Hessians
    Hf0 = zeros(n*n);
    A = zeros(n,n); B = zeros(n,n);

    for i = 1:n*n
    for j = 1:n*n
        A = zeros(n,n); B = zeros(n,n);
        A(i) = 1; B(j) = 1;
        Hf0(i,j) = trace(inv(Uk)*A*inv(Uk)*B);
    end
    end

    Hphi = zeros(n*n);
    for j = 1:n
        uj = Uk(j,:);
        for i = 1:k
            yi = Y(:,i);            
            HRow = -yi*yi'/(M - uj*yi)^2 - yi*yi'/(M + uj*yi)^2;

            for ii = 0:n-1
            for jj = 0:n-1
                row = j+ii*4; col = j+jj*4;
                Hphi(row,col) = Hphi(row,col) + HRow(ii+1,jj+1);
            end
            end
        end
    end

    Hf = t*Hf0 - Hphi;

    % Compute Newton Steps
    Hf_inv = inv(Hf);
    Unt = -Hf_inv*gf;
    ndec(iter) = sqrt(gf' * Hf_inv * gf);
    tau = 1;
    while (-log(abs(det(Uk))) < -log(abs(det(Uk - tau * reshape(Unt, [n,n])))))
        tau = 0.5 * tau;
    end
    Uk = Uk - tau * reshape(Unt, [n,n]);
    res(iter) = norm(Uk - inv(H), 'fro'); 
end

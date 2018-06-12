% Numerical Gradient Testing
% May 2018
%
% Tests implementation of objective and constraint graidents
%
clear all; close all;

disp('Running symbolic gradident checks...');

%% Graident of f0
n = 4;
U = sym('u', [n,n], 'real');
f = -log(abs(det(U)));
df = gradient(f, reshape(U, [n*n,1]));

grad_f0_sym = reshape(df, [n,n]);
grad_f0 = - inv(U)';

% 10 random tests to check if custom implementation is the same as matlab's
num_checks = 10;
check = NaN(num_checks,1);
for i = 1:10
    Utest = rand(n,n);
    gradU = double(subs(grad_f0, U, Utest));
    gradU_sym = double(subs(grad_f0_sym, U, Utest));
    check(i) = norm(gradU - gradU_sym, 'fro');
end
disp(['Maximum grad_f0 difference: ' num2str(max(check))]);


%% Hessian of f0
Hf0_sym = hessian(f, reshape(U, [n*n,1]));
Hf0 = sym('Hf0', [n*n,n*n]);
A = zeros(n,n); B = zeros(n,n);

% Construct Hessian
for i = 1:n*n
    for j = 1:n*n
        A = zeros(n,n); B = zeros(n,n);
        A(i) = 1; B(j) = 1;
        Hf0(i,j) = trace(inv(U)*A*inv(U)*B);
    end
end

check_Hf0 = NaN(num_checks,1);
for i = 1:10
    Utest = rand(n,n);
    Hf0_sym_eval = double(subs(Hf0_sym, U, Utest));
    Hf0_eval = double(subs(Hf0, U, Utest));
    check_Hf0(i) = norm(Hf0_eval - Hf0_sym_eval, 'fro')
end
disp(['Maximum Hf0 difference: ' num2str(max(check_Hf0))]);


%% Graident of constraints
M = 1;          %BPSK
k = 10;         %10 symbols

%Random square channel
H = randn(n);

%Make symbols
X = 2*randi(M+1,n,k) - 3;

%Test with no noise 
Y = H*X;

uj = sym('uj', [1,n], 'real');
yi = sym('yi', [n,1], 'real');
phi = log(M - uj*yi) + log(M + uj*yi);
grad_phi_sym = gradient(phi, uj)';
grad_phi = -yi/(M - uj*yi) + yi/(M + uj*yi);

% Checks
check_grad_phi = NaN(num_checks,1);
for c = 1:num_checks
    grad_phi_test = zeros(n,n);
    grad_phi_sym_test = zeros(n,n);
    Utest = rand(n,n);
    
    for j = 1:n
        uj_test = Utest(j,:);
        for i = 1:k
            yi_test = Y(:,i);
            gradRow = double(subs(subs(grad_phi, uj, uj_test), yi, yi_test))';
            gradRow_sym = double(subs(subs(grad_phi_sym, uj, uj_test), yi, yi_test));
            
            grad_phi_test(j,:) =  grad_phi_test(j,:) + gradRow;
            grad_phi_sym_test(j,:) =  grad_phi_sym_test(j,:) + gradRow_sym;
            
        end
    end
    
    check_grad_phi(c) = norm(grad_phi_test - grad_phi_sym_test, 'fro');
end
disp(['Maximum grad_phi difference: ' num2str(max(check_grad_phi))]);


%% Hessian of constraints
Hphi_sym = hessian(phi, uj);
Hphi = -yi*yi'/(M - uj*yi)^2 - yi*yi'/(M + uj*yi)^2;

%Checks
check_Hphi = NaN(num_checks,1);
for c = 1:num_checks
    Hphi_test = zeros(n*n);
    Hphi_sym_test = zeros(n*n);
    Utest = rand(n,n);
    
    for j = 1:n
        uj_test = Utest(j,:);
        for i = 1:k
            yi_test = Y(:,i);
            
            HRow = double(subs(subs(Hphi, uj, uj_test), yi, yi_test));
            HRow_sym = double(subs(subs(Hphi_sym, uj, uj_test), yi, yi_test));
            
            for ii = 0:n-1
                for jj = 0:n-1
                    row = j+ii*4; col = j+jj*4;
                    Hphi_test(row,col) = Hphi_test(row,col) + HRow(ii+1,jj+1);
                    Hphi_sym_test(row,col) = Hphi_sym_test(row,col) + HRow_sym(ii+1,jj+1);      
                end
            end
        end
    end
    check_Hphi(c) = norm(Hphi_test - Hphi_sym_test, 'fro');
end
disp(['Maximum Hphi difference: ' num2str(max(check_Hphi))]);

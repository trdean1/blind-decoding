clear all; close all;

% Purpose of script: this uses test_single_run as a basis and simply loops 
% for a single randomply generated gain matrix H and noise, varying the
% number of permutations used. It plots the errors discussed for
% test_single_run against the number of permutations used, as well as the
% difference between these errors.
% 
% NOTE 1: This code is inefficient and
% recomputes the results for permutations, but since it was just a proof
% of concept and the runtime wasn't horrific, we did not worry about this.
% 
% NOTE 2: As discussed in the report and paper, the given in those two
% documents is the result of running this script multiple times for
% different rng seeds and averaging the results.

rng(3)
n = 4;
m = n+2;
k = n+1;
maxPerm = 15;
pamSize = 2;
noiseLevel = 1; % 0 = noiseless

%Random square channel
H = randn(m,n);

%Make symbols
X = [1  1  1  1  1;...
    1  1 -1 -1  1;...
    1 -1  1 -1  1;...
    1 -1 -1  1 -1];

%Test with no noise
Y = H*X + noiseOn*(1e-2*randn(m,k));
rm = randn(n);
[q,r] = qr(rm);
x0 = 0.1*reshape(q,[n^2,1]);
poss_perm = combnk(1:m,n);
cB = pamSize*ones(2*n*k,1)-1;
obj = @det_from_list;

options = optimoptions('fmincon','MaxFunctionEvaluations',25000,'OptimalityTolerance',1e-8, 'StepTolerance',1e-7,'Display','off');
for numPerm = 1:maxPerm
    if mod(numPerm,10) == 0
        numPerm
    end
    H_perm = zeros(n,n,numPerm);
    U_perm = zeros(n,n,numPerm);
    T = zeros(n,n,numPerm);
    A_perm = zeros(n,n,numPerm);
    err_perm = zeros(1,numPerm);
    is_atm = zeros(1,numPerm);
    A = zeros(size(H));
    count = zeros(size(H));
    count_for_avg = 0;

    for i = 1:numPerm
        curr_perm = poss_perm(i,:);
        H_perm(:,:,i) = H(curr_perm,:);
        
        cA = construct_constraints(transpose(Y(curr_perm,:)));
        
        U_flat = fmincon(obj,x0,cA,cB,[],[],[],[],[],options);
        U_perm(:,:,i) = transpose(reshape(U_flat,[n,n]));
        T(:,:,i) = U_perm(:,:,i)*H_perm(:,:,i);
        is_atm(i) = isATM(T(:,:,i));
        if is_atm(i)
            A_perm(:,:,i) = inv(U_perm(:,:,i)) * round(T(:,:,i));
            A(curr_perm,:) = A(curr_perm,:) + A_perm(:,:,i);
            count(curr_perm,:) = count(curr_perm,:) + 1;
            err_perm(i) = norm(A_perm(:,:,i) - H_perm(:,:,i),'fro');
            count_for_avg = count_for_avg + 1;
        end
    end
    A = A ./ count;
    err(numPerm) = norm(A(count ~= 0) - H(count ~= 0),'fro')/(n*(sum(count(:,1) ~= 0)));
    err_avg(numPerm) = sum(err_perm)/(count_for_avg * n^2);
end

figure(1)
plot(1:maxPerm,err, 1:maxPerm,err_avg, 'LineWidth',2);
xlabel('Allowed Permutations', 'Interpreter', 'latex')
title('Error for decoded channes of gain matrix', 'Interpreter', 'latex')

figure(2)
plot(1:maxPerm,err-err_avg, 'LineWidth',2);

function [ A ] = construct_constraints( y )
%Uses y to construct a series of contraints corresponding
%to constraining the l-infinity norm of U*Y
s = size(y);
n = s(2);
samples = s(1);
A = zeros(2*samples,2*n);
for i = 1:samples
    for j = 1:n
        for k = 1:n
            A(2*n*(i-1) + j, (j-1)*n+k)  = y(i,k);
            A(2*n*(i-1) + n + j,(j-1)*n+k ) = -1*y(i,k);
        end
    end
end
end

function [ f,gradf ] = det_from_list( x )
%Return the objective function and gradient
%there is some scaling here to make fmincon work better
s = size(x);
n = sqrt(s(1));
m = zeros(n,n);
m=reshape(x,[n,n]);
f = -5000000*log(abs(det(m)));
gradf = reshape(-1*5000000*inv(m)',[n^2,1]);
end
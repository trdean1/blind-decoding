clear all; close all;

% Purpose of script: this was an initial testing script to which was then
% generalized to test_perm_loop. The algorithm is described explicitly in
% the report, so we will only give a quick outline here.
% We randomly generate a gain matrix H, define an input X to ensure 
% recoverability (eg full rank and few other conditions specified in 
% Dean et al), add noise (if noiseLevel != 0), and attempt to revoer H_perm
% for a variety of permutations of Y, called Y_perm. The resultant
% "recovered" H_perm's, called A_perm in the code, are then averaged to a 
% final estimate A. The error of this estimate, measured as the frobenius
% norm of the difference from the ground-truth H, was then compared with
% the average errors of all the A_perms.

rng(0)
n = 4;
m = n+5;
k = n+1;
numPerm = 10%126;
pamSize = 2;
noiseLevel = 1; % 0 = noiseless

%Random square channel
H = randn(m,n);

%Make symbols -- we choose this X to ensure recoverability
X = [1  1  1  1  1;...
        1  1 -1 -1  1;...
        1 -1  1 -1  1;...
        1 -1 -1  1 -1];

%Test with no noise
Y = H*X + noiseLevel*(1e-2*randn(m,k));
rm = randn(n); % for generating first guess x0
[q,r] = qr(rm);
x0 = 0.1*reshape(q,[n^2,1]);
poss_perm = combnk(1:m,n);
cB = pamSize*ones(2*n*k,1)-1;
obj = @det_from_list;

options = optimoptions('fmincon','MaxFunctionEvaluations',25000,'OptimalityTolerance',1e-8, 'StepTolerance',1e-7,'Display','off');

% Note that it is actually unnecessary to store most of the following variables;
% however, it was helpful in debugging and understanding what was going on.
H_perm = zeros(n,n,numPerm);
U_perm = zeros(n,n,numPerm);
T = zeros(n,n,numPerm);
A_perm = zeros(n,n,numPerm);
err_perm = zeros(1,numPerm);
is_atm = zeros(1,numPerm);
A = zeros(size(H));
count = zeros(size(H));

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
    end
end
A = A ./ count;
err = norm(A(count ~= 0) - H(count ~= 0),'fro')/(n*(sum(count(:,1) ~= 0)))
err_avg = mean(err_perm)/(n^2)

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
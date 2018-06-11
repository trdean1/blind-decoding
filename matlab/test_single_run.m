clear all; close all;

rng(0)
n = 4;
m = n+5;
k = n+1;
numPerm = 10%126;
pamSize = 2;
noiseOn = 1;

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
j=0;
poss_perm = combnk(1:m,n);
cB = pamSize*ones(2*n*k,1)-1;
obj = @det_from_list;

options = optimoptions('fmincon','MaxFunctionEvaluations',25000,'OptimalityTolerance',1e-8, 'StepTolerance',1e-7,'Display','off');

H_perm = zeros(n,n,numPerm);
U_perm = zeros(n,n,numPerm);
T = zeros(n,n,numPerm);
A_perm = zeros(n,n,numPerm);
err_perm = zeros(1,numPerm);
is_atm = zeros(1,numPerm);
A = zeros(size(H));
count = zeros(size(H));
err_cum = zeros(size(H));

%disp(['length numPerm = ' num2str(length(numPerm))]);
for i = 1:numPerm
    curr_perm = poss_perm(i,:);
    H_perm(:,:,i) = H(curr_perm,:);
    
    %x0 is a random unitary matrix
    %         rm = randn(n);
    %         [q,r] = qr(rm);
    %         x0 = 0.1*reshape(q,[n^2,1]);
    
    %cA and cB are the contraints passed to fmincon
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
        err_cum(curr_perm,:) = abs(A_perm(:,:,i) - H_perm(:,:,i));
    end
%     if is_atm(i)%abs(det(U*H_perm(:,:,i)))-1.0 < 0.01
%         j = i;
%         break;
%     end
end
A = A ./ count;
is_atm
err = norm(A(count ~= 0) - H(count ~= 0),'fro')/(n*(sum(count(:,1) ~= 0)))
err_avg = mean(err_perm)/(n^2)
err_del_avg = norm(err_cum(count ~= 0),'fro')/(n*(sum(count(:,1) ~= 0)))

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
% function [xopt] = newton_cent(Y, M, u0, num_steps, alpha, beta)
% Infeasible start Newton method for LP centering problem:
%   minimize    cTx - sum(log xi)
%   subject to  Ax = b
%
% Inputs: A, b, c, x0, alpha, beta
%   x0 > 0
%   A: m x n, m < n (fat, full rank).
%   alpha in (0, 0.5), beta in (0, 1)
% Output: xopt (primal opt pt), vopt (dual opt pt), num_newton_steps
%
% Stopping criteria: 
%   1. L2 norm of primal and dual residuals ||r(x,v)||2 < 1e-6
%   2. maximum number of iterations iter_max reached
clear all; close all;
% tol = 1e-6;
% n = size(Y,1);
% k = size(Y,2);
n=4;

% U = reshape(ut,[n,n]);
xopt = 0;

Usym = sym('usym', [n,n], 'real');
grad_f0 = reshape(-inv(Usym)', [n*n,1]);
Hf0 = sym('Hf0', [n*n,n*n]);
A = zeros(n,n); B = zeros(n,n);

for i = 1:n*n
    for j = 1:n*n
        A = zeros(n,n); B = zeros(n,n);
        A(i) = 1; B(j) = 1;
        Hf0(i,j) = trace(inv(Usym)*A*inv(Usym)*B);
    end
end

% Hf0_inv = inv(Hf0);

%% 
for iter = 1:num_steps
    
    % Gradient
    gf0k = sub(grad_f0, Usym, Uk);
    
    
    
    %Hessian
    
    Hf0 = 0 %matrix cookbook (60)
    
    
    
end


% end



% function [xopt,vopt, residuals] = newton_lp_cent(A,b,c,x0,alpha,beta)
% 
% 
% xt = x0;
% vt = zeros(m,1);
% iter = 0;
% residuals = [];
% 
% while true
%     iter = iter + 1;
%     vt1  = vt;
%     
%     %Compute primal and dual Newton Steps
%     H      = diag(xt.^-2);
%     Hinv   = diag(xt.^2);
%     g      = c - 1./xt;   
%     h      = A*xt - b;
%     
%     % Algorithm 10.3
%     HinvAT = Hinv*A';
%     Hinvg  = Hinv*g;
%     
%     Scomp  = -A*HinvAT;
%     vt     = Scomp \ (A*Hinvg - h);
%     xnt    = H     \ (-A'*vt - g);
% 
%     vnt    = vt - vt1;
%     
%     %Backtracking line search on ||res||
%     t = 1;
%     while (res(xt + t*xnt, vt + t*vnt,A,b,c) > (1-alpha*t)*res(xt,vt,A,b,c) || any(xt + t*xnt <= 0)) && t > tol
%         t = beta*t;
%     end
%     
%     % Update only if step "worked"
%     if t > tol
%         xt = xt + t*xnt;
%         vt = vt + t*vnt;
%     end
% 
%     res_t = res(xt, vt,A,b,c);
%     residuals = [residuals; res_t];
%     
%     if ( all(A*xt - b <= tol) && res_t <= tol) || iter >= max_iter
%         xopt = xt;
%         vopt = vt;
%         if iter >= max_iter
%             disp('Maximum number of iterations reached') 
%             disp('Problem likely infeasible or unbounded below');
%         end
%         break;
%     end
% end
% % end
% 
% 
% function residual = res(x,v,A,b,c)
%     r_prim = A*x - b;
%     r_dual = c - 1./x + A'*v;
%     residual = norm([r_prim; r_dual]);
% end



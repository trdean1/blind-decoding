clear all; close all;
n = 4;
M = 1;          %BPSK
k = 6;          %num symbols

%Random square channel
H = randn(n);

%Make symbols
X = 2*randi(M+1,n,k) - 3;
M = M + 0.5;

%Test with no noise 
Y = H*X;

% Construct B from Y
B = [];
for i=1:n
    B = blkdiag(B, Y');
end
clearvars H X Y i

C           = [ B; 
               -B];
b           = ones(2*n*k,1) * M;
x0          = randn(n*n,1);
w0          = C*x0 + b;
w0(w0<0)    = 1;
alpha       = 0.1;
beta        = 0.9;


% Infeasible start Newton method for LP centering problem:
%   minimize    -sum(log wi)
%   subject to  w = b - Ax
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

tol = 1e-6;
max_iter = 5;
% n = length(c);
% m = length(b);

if ~all(w0); disp('Error: x0 must be positive'); return; end

wt = w0;
xt = x0;
vt = zeros(2*n*k,1);
iter = 0;
residuals = [];
rps = [norm(wt + C*xt - b)];

while true
    iter = iter + 1;
    
    %Compute primal and dual Newton Steps
    %Graident and Hessian
    H      = diag(wt.^-2);
%     Hinv   = diag(wt.^2);
    g      = -1./wt;   
%     h      = C*xt - b;

    %Residual
    rp = wt + C*xt - b;
    
    % accpm slide 6-7    
    S       = -C'*H*C;
    dx      = S \ (C'*(g - H*rp));
    dw      = -C*dx - rp;
    dv      = -H*dw - g - vt;
%     vt     = S \ (C*Hinvg - h);
%     xnt    = H     \ (-C'*vt - g);
%     vnt    = vt - vt1;
    
    %Backtracking line search on ||res||
    t = 1;
    while any(wt + t*dw <= 0) && t > tol
        t = beta*t;
    end
    while res(xt + t*dx, wt + t*dw, vt + t*dv,C,b,g) > (1-alpha*t)*res(xt,wt,vt,C,b,g) && t > tol
        t = beta*t;
    end
    
    % Update only if step "worked"
    if t > tol
        disp(num2str(iter));
        xt = xt + t*dx;
        wt = wt + t*dw;
        vt = vt + t*dv;
    end

    res_t = res(xt,wt,vt,C,b,g);
    residuals = [residuals; res_t];
    rps = [rps; norm(wt + C*xt - b)];
    
    if ( all(rp <= tol) && res_t <= tol) || iter >= max_iter
        xopt = xt;
        vopt = vt;
        if iter >= max_iter
            disp('Maximum number of iterations reached') 
        end
        break;
    end
end
rps


function residual = res(x,w,v,C,b,g)
    r_prim = w - C*x - b;
    r_dual = [C'*v; g + v];
    residual = norm([r_prim; r_dual]);
end
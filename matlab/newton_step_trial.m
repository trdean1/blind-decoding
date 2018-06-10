function [U, dists, objs, residuals, rps, rds] = newton_step_trial(n, k, alpha, beta, tau, max_iter, scale_factor)
% Infeasible start Newton method for LP centering problem:
%   minimize    -sum(log wi)
%   subject to  w = b - Cx
%
% Inputs: TODO
%   x0 > 0
%   A: m x n, m < n (fat, full rank).
%   alpha in (0, 0.5), beta in (0, 1)
% Output: xopt (primal opt pt), vopt (dual opt pt), num_newton_steps
%
% Stopping criteria: 
%   1. L2 norm of primal and dual residuals ||r(x,v)||2 < 1e-6
%   2. maximum number of iterations iter_max reached
tol = 1e-6;


%Random square channel
randn('state', 0);
channel = randn(n);

%Make symbols
M = 1;          %BPSK
X = 2*randi(M+1,n,k) - 3;
M = M*scale_factor;

%Test with no noise 
Y = channel*X;

% Construct B from Y
B = [];
for i=1:n
    B = blkdiag(B, Y');
end

C           = [ B; 
               -B];
b           = ones(2*n*k,1) * M;
x0          = randn(n*n,1);
w0          = b - C*x0;  %w0          = C*x0 + b;
w0(w0<0)    = 1;


wt = w0;
xt = x0;
vt = zeros(2*n*k,1);
iter = 0;
residuals = [];
rps = [];
rds = [];
objs = [];
dists = [];

while true
    iter = iter + 1;
    
    %Compute primal and dual Newton Steps
    %Graident and Hessian
    invX     = inv(reshape(xt, [n,n]));
    H_x      = zeros(n*n);
    for i = 1:n*n
    for j = 1:n*n
        A = zeros(n,n); B = zeros(n,n);
        A(i) = 1; B(j) = 1;
        H_x(i,j) = trace(invX*A*invX*B);
    end
    end
    
    H_w      = 1/tau * diag(wt.^-2);
    g_w      = 1/tau * -1./wt;
    g_x      = reshape(-invX, [n*n,1]);

    %Residuals
    res_t = res(xt,wt,vt,C,b,g_w,g_x);
    residuals = [residuals; res_t];
    rp = wt + C*xt - b;
    rd = [g_x + C'*vt; g_w + vt];
    rps = [rps; norm(rp)];
    rds = [rds; norm(rd)];
    
    % Objective and distance to opt
    Ut = reshape(xt, [n,n])';
    objs = [objs -log(abs(det(Ut)))];
    UY = Ut*Y;
    dists = [dists norm(UY - sign(UY), 'fro')];
    
    % accpm slide 6-7    
%     S       = C'*H_w*C;               % sign changed based on posted code
%     dx      = S \ (C'*(g_w - H_w*rp));
%     dw      = -C*dx - rp;
%     dv      = -H_w*dw - g_w - vt;
    kkt_mat = [H_x              zeros(n*n,2*n*k)    C';
               zeros(2*n*k,n*n) H_w                 eye(2*n*k);
               C                eye(2*n*k)          zeros(2*n*k)];
    kkt_sol = kkt_mat \ -[rd; rp];
    dx = kkt_sol(1:n*n);
    dw = kkt_sol(n*n+1: 2*n*k + n*n);
    dv = kkt_sol(2*n*k + n*n + 1 : end);
           
    %Backtracking line search on ||res||
    t = 1;
    while any(wt + t*dw <= 0) && t > tol
        t = beta*t;
    end
    while res(xt + t*dx, wt + t*dw, vt + t*dv,C,b,g_w,g_x) > (1-alpha*t)*res(xt,wt,vt,C,b,g_w,g_x) && t > tol
        t = beta*t;
    end
    
    % Update only if step "worked"
    if t > tol
%         disp(num2str(iter));
        xt = xt + t*dx;
        wt = wt + t*dw;
        vt = vt + t*dv;
    end
    
    if ( all(rp <= tol) && res_t <= tol) || iter >= max_iter
        xopt = xt;
        vopt = vt;
%         if iter >= max_iter
%             disp('Maximum number of iterations reached') 
%         end
        break;
    end
end

% Compute final residuals, objective & distances
res_t = res(xt,wt,vt,C,b,g_w,g_x);
residuals = [residuals; res_t];
rps = [rps; norm(wt + C*xt - b)];
rds = [rds; norm([C'*vt; g_w + vt])];

U = reshape(xt, [n,n])';
objs = [objs -log(abs(det(U)))];
UY = U*Y;
dists = [dists norm(UY - sign(UY), 'fro')];


end


function residual = res(x,w,v,C,b,gw,gx)
    r_prim = w + C*x - b;
    r_dual = [gx + C'*v; gw + v];
    residual = norm([r_prim; r_dual]);
end
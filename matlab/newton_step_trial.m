function [U, dists, mindists, objs, residuals, rps, rds] = newton_step_trial(n, k, alpha, beta, tau, max_iter, M, Y)
% 
% Find a matrix U such that U*Y is near a vertex.  
% Our goal is to minimize -log|det U| subject to norm(U*Y,inf) <= 1.  While
% the problem is non-convex, solutions occur on the vertices of the 
% feasible region, i.e. U*Y is in {+1,-1}^(nxk).  
% Thus, we reformulate the problem, add a log barrier, and use an 
% infeasible start Newton method with the goal of getting U*Y close 
% to a vertex.  
%
% Inputs:
%   n: Size of square channel matrix
%   k: Number of received symbols.
%   alpha in (0, 0.5), beta in (0, 1), used for backtracking line search
%   tau: Initial weightage on barrier is 1/tau
%   M: Should equal 1 for BPSK.
%   Y: Matrix of received symbols.
%
% Outputs:
%   U: Matrix such that U*Y is hopefully near a vertex
%   dists: Distances of U*Y to vertex on each iteration
%   mindists:  Minimum distances so far
%   objs:  Value of objective at each iteration
%   residuals: Vector of norm(rp,rd) per iteration
%   rps: Primal residuals
%   rds: Dual residuals
%
% Stopping criteria:
%   1. L2 norm of primal and dual residuals ||r(x,v)||2 < 1e-6
%   2. maximum number of iterations iter_max reached
tol = 1e-6;

% Construct B from Y
B = [];
for i=1:n
    B = blkdiag(B, Y');
end

% Problem formulation as described in our report
C           = [ B;
    -B];
b           = ones(2*n*k,1) * M;
x0          = randn(n*n,1);
w0          = b - C*x0;
w0(w0<0)    = 1;

% Initialization
wt = w0;
xt = x0;
vt = zeros(2*n*k,1);
iter = 0;
residuals = [];
rps = []; %Norms of primal residuals
rds = []; %Norms of dual residuals
objs = []; %Values of objective
dists = []; %Distance to nearest vertex
mindists = []; % Minimum distance so far

while true
    iter = iter + 1;
    
    %Compute primal and dual Newton Steps
    %Gradient and Hessian
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
    
    % Objective and distance to vertex
    Ut = reshape(xt, [n,n])';
    objs = [objs -log(abs(det(Ut)))];
    UY = Ut*Y;
    dists = [dists norm(UY - sign(UY), 'fro')];
    if (iter == 1)
        mindists = [dists(1)];
    else
        if(dists(iter) < mindists(iter-1))
            mindists = [mindists dists(iter)];
        else
            mindists = [mindists mindists(iter-1)];
        end
    end
    
    % Solve our KKT system to find dw, dx, and dv
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
        xt = xt + t*dx;
        wt = wt + t*dw;
        vt = vt + t*dv;
    end
    
    % Decrease barrier weightage is distance does not decrease much
    if (iter > 1 && dists(iter-1) - dists(iter)<10^-2)
        tau = tau*10 ;
    end
    
    if ( all(rp <= tol) && res_t <= tol) || iter >= max_iter
        xopt = xt;
        vopt = vt;
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
if(dists(end) < mindists(end))
    mindists = [mindists dists(end)];
else
    mindists = [mindists mindists(end)];
end

end


function residual = res(x,w,v,C,b,gw,gx)
    r_prim = w + C*x - b;
    r_dual = [gx + C'*v; gw + v];
    residual = norm([r_prim; r_dual]);
end

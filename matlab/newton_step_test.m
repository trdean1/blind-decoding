% Newton Step Testing:  Determining if our method produces a U such that
% U*Y is near a vertex.  Vertices occur when U*Y = Xhat has elements 
% that are only +-1.  Calls newton_step_trial.m for each test.
% Plots minimum distance over iteration for primal feasible runs.
clear all; close all;
randn('state', 0);

% Testing params
ntrial = 50;
ks = [10, 12, 15];
ns = 4:8;

rprim_zero = zeros(length(ks), length(ns));
for kind = 1:length(ks)
for nind = 1:length(ns)
    
% Problem params
n       = ns(nind);
k       = ks(kind);

% Solver params
alpha   = 0.1;
beta    = 0.9;
tau     = 1e-2;
max_iter= 5;

num_sucessful = 0;
num_recovered = 0;
distances = zeros(ntrial, max_iter+1);
min_distances = zeros(ntrial, max_iter+1);
r_primals = zeros(ntrial, max_iter+1);
objective = zeros(ntrial, max_iter+1);
for trial = 1:ntrial
    %Random square channel
    H = randn(n);

    %Make symbols
    M = 1;          %BPSK
    X = 2*randi(M+1,n,k) - 3;
    
    %Test with no noise 
    Y = H*X;
    
    [U, dists, mindists, objs, residuals, rps, rds] = ...
            newton_step_trial(n, k, alpha, beta, tau, max_iter, M, Y);
    distances(trial,:) = dists;
    min_distances(trial,:) = mindists;
    r_primals(trial,:) = rps;
    r_duals(trial,:) = rds;
    objective(trial,:) = objs;

    % Check primal feasibility
    if abs(r_primals(trial,end)) < 1e-3
        num_sucessful = num_sucessful + 1;
    end
    
%     % Check ATM
%     vert = sign(U*Y);
%     if isPermute(vert, X)
%         num_recovered = num_recovered + 1;
%     end
    
end
fprintf('\n%d/%d', (kind-1)*length(ks) + nind, length(ks)*length(ns));
rprim_zero(kind, nind) = num_sucessful;

end
end

%% Plotting
figure;
for trial = 1:ntrial
    if abs(r_primals(trial,end)) < 1e-3
        semilogy(0:max_iter, min_distances(trial,:), 'linewidth', 2);
        hold on;
    end
end
set(gca, 'fontsize',14);
title('Distance to vertex vs iteration', 'interpreter', 'latex');
ylabel('Distance to vertex ($l_2$)', 'interpreter', 'latex');
xlabel('Iteration', 'interpreter', 'latex');
grid on;

%%
figure; 
subplot(1,2,1);
hold on;
for trial = 1:ntrial
%     if abs(r_primals(trial,end)) < 1e-3
        plot(0:max_iter, r_primals(trial,:), 'linewidth', 2);
%     end
end
set(gca, 'fontsize',14);
title('Primal Residuals', 'interpreter', 'latex');
ylabel('Residual ($l_2$)', 'interpreter', 'latex');
xlabel('Iteration', 'interpreter', 'latex');
grid on;

subplot(1,2,2);

for trial = 1:ntrial
%     if abs(r_primals(trial,end)) < 1e-3
        semilogy(0:max_iter, r_duals(trial,:), 'linewidth', 2);
        hold on;
%     end
end
set(gca, 'fontsize',14);
title('Dual Residuals', 'interpreter', 'latex');
ylabel('Residual ($l_2$)', 'interpreter', 'latex');
xlabel('Iteration', 'interpreter', 'latex');
grid on;

% figure; hold on;
% for trial = 1:ntrial
%     if abs(r_primals(trial,end)) < 1e-3
%         num_sucessful = num_sucessful + 1;
%         plot(0:max_iter, objective(trial,:), 'linewidth', 2);
%     end
% end
% set(gca,'fontname','arial','fontsize',14);
% title('Objective vs iteration');
% ylabel('objective');
% xlabel('Iteration');

%% 
figure; hold on;
for kind = 1:length(ks)
    plot(ns, 2*rprim_zero(kind,:), 'linewidth', 3, 'displayname', ['k = ' num2str(ks(kind))]);
end
set(gca, 'fontsize',14);
title('Percent of primal feasible runs after 5 steps vs $n$', 'interpreter', 'latex');
ylabel('Percent of primal feasible $U$s', 'interpreter', 'latex');
xlabel('n', 'interpreter', 'latex');
grid on;
legend('show');
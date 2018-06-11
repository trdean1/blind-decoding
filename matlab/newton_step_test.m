% Newton Step Testing
clear all; close all;
randn('state', 0);

% Testing params
ntrial = 50;

% Problem params
n       = 5;
k       = 12;
scale   = 1;

% Solver params
alpha   = 0.1;
beta    = 0.9;
tau     = 1e-2;
max_iter= 50;

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
    
    % Check ATM
    vert = sign(U*Y);
    
    
end


%% Plotting
num_sucessful = 0;
figure;
for trial = 1:ntrial
    if abs(r_primals(trial,end)) < 1e-3
        num_sucessful = num_sucessful + 1;
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
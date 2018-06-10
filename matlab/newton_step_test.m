% Newton Step Testing
clear all; close all;

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
max_iter= 5;

distances = zeros(ntrial, max_iter+1);
min_distances = zeros(ntrial, max_iter+1);
r_primals = zeros(ntrial, max_iter+1);
objective = zeros(ntrial, max_iter+1);
for trial = 1:ntrial
    [U, dists, mindists, objs, residuals, rps, rds] = ...
            newton_step_trial(n, k, alpha, beta, tau, max_iter, scale);
    distances(trial,:) = dists;
    min_distances(trial,:) = mindists;
    r_primals(trial,:) = rps;
    objective(trial,:) = objs;
end


%% Plotting
num_sucessful = 0;
figure; hold on;
for trial = 1:ntrial
    if abs(r_primals(trial,end)) < 1e-3
        num_sucessful = num_sucessful + 1;
        plot(0:max_iter, min_distances(trial,:), 'linewidth', 2);
    end
end
set(gca, 'fontsize',14);
title('Distance to vertex vs iteration', 'interpreter', 'latex');
ylabel('Distance to vertex ($l_2$)', 'interpreter', 'latex');
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
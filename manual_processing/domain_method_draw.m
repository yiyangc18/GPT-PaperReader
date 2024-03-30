% Most Recent Data Preparation
algorithms = {'SAC', 'PPO', 'TD3', 'DQN', 'DDPG', 'Actor Critic', 'PG', 'Model-based DRL', 'ADP'};
problems = {'Path Planning', 'Lateral Control', 'Longitudinal Control', 'Longitudinal and Lateral Control', 'End to End'};
data_transposed = [
    3, 3, 4, 4, 5, 1, 0, 0, 0; % Path Planning
    0, 1, 0, 1, 3, 2, 1, 0, 0; % Lateral Control
    0, 0, 0, 0, 3, 0, 0, 1, 1; % Longitudinal Control
    0, 0, 0, 0, 1, 0, 1, 1, 1; % Longitudinal and Lateral Control
    1, 1, 0, 1, 1, 2, 1, 1, 0  % End to End
]';

% Plotting
figure;
b = bar(data_transposed, 'stacked');
set(gca, 'XTickLabel', algorithms);
legend(problems, 'Location', 'bestoutside', 'Interpreter', 'none');
ylabel('Number of Papers');
title('Latest Distribution of RL Algorithms across Different Problems');

% Beautify the graph
set(gca, 'XTickLabelRotation', 45);
grid on;
box on;
set(gcf, 'Position', [100, 100, 1200, 600]); % Adjust figure size

%%

domains = ["Behavior Decision", "Traffic Efficiency", "Motion Planning", "V2X Communication", ...
           "Lateral Control", "Energy Management", "Velocity Control", "Trajectory Optimization", ...
           "Dynamic Obstacle Avoidance", "Risk Assessment"];
counts = [139, 128, 74, 59, 40, 19, 18, 12, 9, 6];

figure;
bar(counts, 'FaceColor', 'flat');
set(gca, 'XTickLabel', domains, 'XTickLabelRotation', 45);
xlabel('Domain');
ylabel('Count');
title('RL-based Works in Automotive (2016-present)');
grid on;
ylim([0 150]); % Adjust Y-axis maximum to 150

% Add counts above bars
for i = 1:length(counts)
    text(i, counts(i), num2str(counts(i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end


%% Fit difference of gaussian

% X axis
x = [0:0.02:5];
% Simulated signal
S = 0.7 * exp(-(x - 1.5).^2 / 2 / 0.2.^2) - ...
    0.5 * exp(-(x - 2.3).^2 / 2 / 0.3.^2) + ...
    0.4*(rand(1,length(x))-0.5);

% Gaussian Fit function
GaussianFit = @(a,x) (a(1) * exp(-(x - a(2)).^2 / 2 / a(3).^2) - ...
              a(4) * exp(-(x - a(5)).^2 / 2 / a(6).^2));

% Initial Fit
InitialFit = [0.8, 1.4, 0.2, 0.5, 2, 0.4];

% An option to see the iteration progress
Options = statset('Display','iter')

% Actually doing fitting
% Best   Solver  x sig fit_function Init_guess  Option
a_best = nlinfit(x, S, GaussianFit, InitialFit, Options)

% Plot results
figure(1); clf; hold on;
plot(x,S,'DisplayName','Signal')
plot(x, GaussianFit(a_best,x),'r','linewidth',1.2,'DisplayName','Fit')
legend
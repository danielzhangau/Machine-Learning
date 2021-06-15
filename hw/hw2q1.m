close all; clc; clear all;

f = @(x) 2.76*cos(5*x);
x = linspace(0, pi, 7)';
y = f(x);
data = [x, y];
save('hw2q1_2020.mat', 'data');
writematrix(data, 'hw2q1_2020.csv');
load('hw2q1_2020.mat')
x = data(:,1); y = data(:, 2);

% Cubic
[model, gof, alg] = fit(x, y, 'poly3');
gof.rsquare

% 10th order poly can hit all points exactly => best R^2 = 1

% Sine wave
sine = fittype({'cos(5*x)', });
[model, gof] = fit(x, y, sine);
gof.rsquare
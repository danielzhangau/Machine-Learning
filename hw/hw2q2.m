close all; clear all; clc;
%{
% Data generating process
xtrain = 8*rand([100, 1]);
ytrain = 2*sin(1.5*xtrain) + randn([100, 1]);
xvalid = 8*rand([100, 1]);
yvalid = 2*sin(1.5*xvalid) + randn([100, 1]);

datatrain = [xtrain, ytrain];
datavalid = [xvalid, yvalid];
save('hw2q2Training_2020.mat', 'datatrain');
save('hw2q2Validation_2020.mat', 'datavalid');
writematrix(datatrain, 'hw2q2Training_2020.csv');
writematrix(datavalid, 'hw2q2Validation_2020.csv');
%}

load('hw2q2Training_2020.mat')
load('hw2q2Validation_2020.mat')
xtrain = datatrain(:,1); ytrain = datatrain(:,2);
xvalid = datavalid(:,1); yvalid = datavalid(:,2);

figure()
plot(xtrain, ytrain, '*')


order_list = 0:30;
sse_list = [];
features = {};
coeffs = {};
for order = order_list
    features= cat(2, features, ['x^'  num2str(order)]);
    coeffs  = cat(2, coeffs, ['a'  num2str(order)]);
    
    poly    = fittype(features, 'coefficients', coeffs);
    model   = fit(xtrain, ytrain, poly);
    ypred   = feval(model, xvalid);
    sse     = sum( (ypred - yvalid).^2);
    sse_list = [sse_list, sse]; 
end

plot(order_list, sse_list)


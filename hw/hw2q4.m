close all; clear all; clc;

data = csvread('iris.csv');
trainingdata = data(:, [1, 5]);


x_axis = -10:0.01:10;
posteriors1 = [];
posteriors2 = [];
posteriors3 = [];
for x = x_axis
    [~, post] = classifier(trainingdata, 3, x);
    posteriors1 = [posteriors1, post(1)];
    posteriors2 = [posteriors2, post(2)];
    posteriors3 = [posteriors3, post(3)];
end
plot(x_axis, posteriors1, 'b'); hold on
plot(x_axis, posteriors2, 'r')
plot(x_axis, posteriors3, 'g')


function [class, posteriors] = classifier(training_data, k, x, priors)
    % Assume priors are uniform if none given
    if nargin < 4
        priors = ones([1,k])/k;
    end
    means = [];
    vars = [];
    likelihoods = [];
    
    for class = 0:k-1
        ind = training_data(:,2) == class;
        mu = mean(training_data(ind, 1));
        % Note MLE is the biased estimator for variance
        v = var(training_data(ind, 1), 1); 
        likelihood = 1/(sqrt(2*pi*v))*exp(-(x-mu).^2/(2*v));
        
        means = [means, mu];
        vars = [vars, v];
        likelihoods = [likelihoods, likelihood];
    end
    posteriors = likelihoods.*priors/(sum(likelihoods.*priors));
    [~, class] = max(posteriors);
end


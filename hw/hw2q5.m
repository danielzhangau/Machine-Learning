close all; clear all; clc;
n = 1000;
k = 10000;
L = 0.2;

aic = 2*k - 2*log(L)
bic = log(n)*k-2*log(L)

likelihood(aic, bic, n)


function out = likelihood(aic, bic, n)
    k = (aic-bic)/(2-log(n));
    logL = k - aic/2;
    out = exp(logL);
end


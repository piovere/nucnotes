function [ B ] = bval( ga, xs_train, ys_train)
%BVAL Summary of this function goes here
%   Detailed explanation goes here

alphas = bestalpha(ga);

B = (xs_train'*xs_train + diag(alphas.^2))\(xs_train'*ys_train);

end


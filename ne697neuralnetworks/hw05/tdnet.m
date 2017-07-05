function [ net ] = tdnet( neurons )
%tdnet Summary of this function goes here
%   Detailed explanation goes here
net = timedelaynet(1:2, neurons);

%   Mean squared error can't be better than the variance of our noise
%   For an unbiased estimator, mse = variance
net.performFcn = 'mse';
net.trainParam.goal = 0.0030; % calculated as var(noise)

%   Train to 1000 epochs
net.trainParam.epochs = 1000;

%   Train on all of the data
net.divideFcn = 'dividetrain';

end


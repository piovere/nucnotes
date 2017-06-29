% ...find the least number of hidden nodes that trains the network for the
% following combinations
net = feedforwardnet(3);
init(net);

% Train to an average error per pattern of 0.2
net.performFcn = 'mae';
net.trainParam.goal = 0.2;
% MATLAB overrides this, probably due to the sign errors it would
% introduce. Therefore I am training mean squared error to 0.04 instead. I
% KNOW THIS IS NOT EXACTLY THE SAME.
net.performFcn = 'mse';
net.trainParam.goal = 0.04;

% Don't train past 300 epochs
net.trainParam.epochs = 300;

% ...you need to try several different initial conditions (weights and
% biases

% Use two different types of hidden layers.
%   - Use logsig hidden layer and linear output layer
af = 'logsig';
%   - Use hyperbolic tangent hidden layer and linear output layer
% af = 'tansig';
net.layers{1}.transferFcn = af;

% train the network to an average error of <= 0.2
[net, tr] = train(net, x, t);
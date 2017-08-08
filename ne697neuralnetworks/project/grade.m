function [ grade ] = grade( spread, x_train, y_train, x_val, y_val )
%grade Summary of this function goes here
%   Detailed explanation goes here
net = grnet(spread, x_train, y_train);
y_ = net(x_val);
delta = abs(y_val - y_);

grade = norm(delta);

end


function [ pnn ] = pnn( width, x_train, cat_train, x_val, cat_val )
%pnn Summary of this function goes here
%   Detailed explanation goes here
n = newpnn(x_train, cat_train, width);
pnn = confusion(cat_val, n(x_val));

end


function [ grnet ] = grnet( spread, x_train, y_train )

%grnet Summary of this function goes here
%   Detailed explanation goes here
grnet = newgrnn(x_train, y_train, spread);

end


function [ a ] = bestalpha( ga )
%BESTALPHA Summary of this function goes here
%   Detailed explanation goes here
[m, i] = min(ga.score);

a = ga.population(i, :)';

end


function [y] = unscore(x, meanval,stdval)
%	
%  [y] = unscore(x, meanval,stdval)
%
%  Un-scale mean centered unit varianced data.
%  Input the scaled data, mean and standard deviation and 
%  return the unscaled data.

% J. Wesley Hines, The University of Tennessee, 1998
% 6-10-98
% Copyright (c) 

[nrows,ncols]=size(x);

y = x .* (ones(nrows,1)*stdval);  % Un-normalize from unit variance.

y = y + ones(nrows,1)*meanval; 	 % Add back on the mean.


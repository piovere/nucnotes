function [y,meanval,stdval] = zscore1(x, mvin,stdin)
%	
%  [y,meanval,stdval] = zscore1(x, mvin,stdin)
%
%  Mean center the data and scale to unit variance.
%  If number of inputs is one, calculate the mean and standard deviation.
%  If the number if inputs is three, use the given mean and SD.

% J. Wesley Hines, The University of Tennessee, 1998
% 6-10-98
% Copyright (c)
%

[nrows,ncols]=size(x);

if nargin == 1
   meanval = mean(x);  	% calculate mean values
else
   meanval = mvin;	% use previously calculated value
end

y = x - ones(nrows,1)*meanval; 	% subtract off mean 

if nargin == 1
   stdval = std(y);	% calculate the SD
else
   stdval = stdin;	% use previously calculated value
end

y = y ./ (ones(nrows,1)*stdval);  % normalize to unit variance

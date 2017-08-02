function d = unscore( scaled, means, stddevs )
%unscore Restore scaled data
%   Restores data that was scaled for machine learning

% Initialize result matrix
d = ones(size(scaled));

% Find out how many columns in d
c = size(d);
c = c(2);

% multiply each column by the relevant standard deviation
for i = 1:c
    d(:,i) = scaled(:,i) .* stddevs(i) + means(i);
end

end


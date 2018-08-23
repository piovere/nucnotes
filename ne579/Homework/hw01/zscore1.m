function [xs, x_mean, x_std] = zscore1(x)
    s = size(x);
    rows = s(1);
    columns = s(2);
    
    mean_vec = ones(1, columns);
    std_vec = ones(1, columns);
    
    for i = 1:columns
        v = x(:,i);
        mean_val = mean(v);
        std_val = std(v);
        c = (v - mean_val) / std_val;
        x(:,i) = c;
        mean_vec(i) = mean_val;
        std_vec(i) = std_val;
    end
    
    xs = x;
    x_mean = mean_vec;
    x_std = std_vec;
end
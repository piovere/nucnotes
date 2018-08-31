function [x_mcuv, std_x, mean_x] = zscore1(m)
    s = size(m);
    rows = s(1);
    columns = s(2);
    
    mean_vec = ones(1, columns);
    std_vec = ones(1, columns);
    
    for i = 1:columns
        v = m(:,i);
        mean_val = mean(v);
        std_val = std(v);
        c = (v - mean_val) / std_val;
        m(:,i) = c;
        mean_vec(i) = mean_val;
        std_vec(i) = std_val;
    end
    
    x_mcuv = m;
    std_x = std_vec;
    mean_x = mean_vec;
end
function [mu,wt] = Quadrature(N_Angles)

wt = zeros(N_Angles,1);
mu = zeros(N_Angles,1);

if N_Angles == 2
    
    mu(1) = 0.5773502691;
    wt(1) = 1.0000000000;
    
elseif N_Angles == 4
    
    mu(1) = 0.3399810435;
    mu(2) = 0.8611363115;
    wt(1) = 0.6521451549;
    wt(2) = 0.3478548451;
    
elseif N_Angles == 8
    
    mu(1) = 0.1834346424;
    mu(2) = 0.5255324099;
    mu(3) = 0.7966664774;
    mu(4) = 0.9602898564;
    wt(1) = 0.3626837834;
    wt(2) = 0.3137066459;
    wt(3) = 0.2223810344;
    wt(4) = 0.1012285363;
    
elseif N_Angles == 12
    
    mu(1) = 0.1252334085;
    mu(2) = 0.3678314989;
    mu(3) = 0.5873179542;
    mu(4) = 0.7699026741;
    mu(5) = 0.9041172563;
    mu(6) = 0.9815606342;
    wt(1) = 0.2491470458;
    wt(2) = 0.2334925365;
    wt(3) = 0.2031674267;
    wt(4) = 0.1600783286;
    wt(5) = 0.1069393260;
    wt(6) = 0.0471753364;

end

for ii = 1:(N_Angles/2)
    
    mu(ii + N_Angles/2) = -mu(ii);
    wt(ii) = wt(ii)/2;
    wt(ii + N_Angles/2) = wt(ii);
    
end

return

end


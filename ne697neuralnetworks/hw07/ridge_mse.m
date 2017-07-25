function [er] = ridge_mse(lambdas,xn,yn,xt,yt) 
%
%   [yq] = ridge_mse(lambdas,xn,yn,xt,yt)
%   Local Ridge Regression Error Calculation.
%   Calcualtes the regression parameters given training inputs and outputs.
%   Use standardixed data: xn, yn with rows=observations.
%
%   lambdas - single or vector of ridge parameters
%   xn      - training input data matrix
%   yn      - training output data vector
%   xt      - test input data matrix
%   yt      - test output data vector
%
%   er     - mean squared error

% Hines 2006
% University of Tennessee

if length(lambdas)==1;
    lambdas=lambdas*ones(1,size(xn,2));  % allow for single ridge parameter
end   
    
beta=inv(xn'*xn+diag(lambdas.^2))*xn'*yn;
yp=xt*beta;
er=mean(sumsqr(yp-yt)/length(yt)); 

 
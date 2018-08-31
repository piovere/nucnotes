function [x,fval,exitflag,output,population,score] = fifty_cont(nvars)
%% This is an auto generated MATLAB file from Optimization Tool.

%% Start with the default options
options = optimoptions('ga');
%% Modify options setting
options = optimoptions(options,'Display', 'off');
[x,fval,exitflag,output,population,score] = ...
ga(@(x)ridge_mse(x,xs_train,ys_train,xs_test,ys_test),nvars,[],[],[],[],[],[],[],[],options);

clear
close all
clc

% addpath /Users/jamiecoble/'Google Drive'/Documents/Classes/'NE 697'/2017/'NE 671_GA'/mfiles/data/
% addpath /Users/jamiecoble/'Google Drive'/Documents/Classes/'NE 697'/2017/'NE 671_GA'/'Conventional Optimization Examples'/'multivariate ridge'/

load sim.mat
%%

train = Data([1:500 1501:2000 3001:3500 4501:end],:);
test = Data([501:1000 2001:2500 3501:4000],:);
val = Data([1001:1500 2501:3000 4001:4500],:);

x_train = train(:,[1:34 36:end]);
y_train = train(:,35);
x_test = test(:,[1:34 36:end]);
y_test = test(:,35);
x_val = val(:,[1:34 36:end]);
y_val = val(:,35);

%%

% For Ridge Regression, we want to standardize the inputs AND output %
[xs x_m x_std] = zscore1(x_train);
[ys y_m y_std] = zscore1(y_train);
xsT = zscore1(x_test,x_m,x_std); % standardize the test inputs using the same mean/std dev!
xsV = zscore1(x_val,x_m,x_std); % standardize validation inputs using the same mean/std dev!

s = svd(xs);  % singular values of the standardized data


%% We want to test regularization parameters that cover the range of
% singular values, so we'll use alphas in the range of [0.1 1000] (for
% convenience)

% We'll do a cross-validation method

alpha = logspace(-1,3,100);
RMSE_train = nan(size(alpha)); % Mean squared training error for each alpha
RMSE_test = nan(size(alpha)); % Mean squared test error for each alpha
normB = nan(size(alpha)); % size the regression coefficients for each alpha
cond_num = nan(size(alpha));

for ii = 1:numel(alpha)
    B = ridge(ys,xs,alpha(ii).^2);
    yps = xs*B;
    yps = unscore(yps,y_m,y_std);
    RMSE_train(ii) = sqrt(mean((yps-y_train).^2));
    yps = xsT*B; % predictions for test data %
    yps = unscore(yps,y_m,y_std); % un-standardize predictions %
    error = y_test - yps;
    RMSE_test(ii) = sqrt(mean(error.^2));
    e = eig(xs'*xs + alpha(ii).^2*eye(size(xs,2)));
    cond_num(ii) = max(e)/min(e);
end

ind = find(RMSE_test == min(RMSE_test));
a_CV = alpha(ind)
RMSE_min = RMSE_test(ind)
con_min = cond_num(ind)
cond_from_s = (max(s)^2 + a_CV^2)/(min(s)^2 + a_CV^2)

%% Evaluating performance on the validation data

BCV = ridge(ys,xs,a_CV^2);
ypCV = xsV*BCV;
ypCV = unscore(ypCV,y_m,y_std);
RMSE_CV = sqrt(mean((ypCV-y_val).^2))

% for regular linear regression
BLR = regress(ys,xs);
ypLR = xsV*BLR;
ypLR = unscore(ypLR,y_m,y_std);
RMSE_linReg = sqrt(mean((ypLR-y_val).^2))

% Plot number 3
f = figure; 
plot(y_val,'k.')
hold all
plot(ypCV,'r-s')
plot(ypLR,'g')
xlabel('Observation')
ylabel('Output')
title('Plot number 3')
legend('Actual','CV-method','Linear Regression','location','best')

%% What if we use optimization methods? %%

% Objective function is MSE of ridge regression - ridge_mse.m %

options = optimset('plotfcn',{@optimplotx, @optimplotfval},'display','iter');
% start at the alpha value selected by cross validation above % 
[a_ridge,fval,exitflag,output] = fminunc(@(x)ridge_mse(x,xs,ys,xsT,zscore1(y_test,y_m,y_std)),a_CV,options)
B_opt = ridge(ys,xs,a_ridge.^2);
yp_opt = xsV*B_opt;
yp_opt = unscore(yp_opt,y_m,y_std);
RMSE_opt = sqrt(mean((yp_opt-y_val).^2))

figure(f)
plot(yp_opt,'b-o')
legend('Actual','CV-method','Linear Regression','Ridge Reg - opt','location','best')
%% Now let's look at local methods %%
 
% Objective function is MSE of ridge regression - ridge_mse.m

a0 = a_CV*ones(size(xs,2),1);
[a_local,fval,exitflag,output] = fminunc(@(x)ridge_mse(x,xs,ys,xsT,zscore1(y_test,y_m,y_std)),a0,options)

%% 
B_opt2 = (xs'*xs+diag(a_local.^2))\(xs'*ys);
yp_opt2 = xsV*B_opt2;
yp_opt2 = unscore(yp_opt2,y_m,y_std);
RMSE_opt2 = sqrt(mean((yp_opt2-y_val).^2))

figure(f)
plot(yp_opt2,'m')
legend('Actual','CV-method','Linear Regression','Ridge Reg - opt','Local Ridge Reg','location','best')

RMSE_linReg
RMSE_CV
RMSE_opt
RMSE_opt2

%% PCA and Filter Factors

% We can compare correlation coefficients of the PC scores to the output
% with the filter factors to get an idea of whether useful information is
% being passed to the model. Correlations with magnitude greater than 0.3
% are useful, greater than 0.7 are very good. Correlations with magnitude
% less than 0.3 are considered poor. High filter factors (close to 1) mean
% that component is being passed to the model. Low filter factors (close to
% 0) mean that component is being filtered out.

% to find the PC scores, we first need to do principal component analysis
% of the training data
[loadings latent perExp] = pcacov(cov(xs)); % loadings is the matrix of PC 
    % loadings (eigenvectors of the covariance matrix), latent is the 
    % corresponding eigenvalues, and perExp gives the percent of variance 
    % explained by each PC. 
    
scores = xs*loadings; % these are the PC scores of the training data.

cc = corrcoef([scores ys]); % correlation matrix. the last column is the 
    % correlation of the scores to the output (expect the n,n position,
    % which is trivially 1.0 - the correlation of y to y)
    
% Recall from line 70, we already calculated the singular values of the 
    % training data. Note that these are related to the eigenvalues we
    % found in line 182.
    
ff_ridge = s.^2./(s.^2 + a_ridge.^2);
ff_local = s.^2./(s.^2 + a_local.^2);

figure;
plot(abs(cc(end,1:end-1)),'o-')
hold on
plot(ff_ridge,'x-')
plot(ff_local,'s-')
legend('Correlation to output','Ridge Filter Factor','Local Filter Factor','location','best')

%% We want the filter factors to roughly match the magnitude of the 
% correlation coefficients .. what if we start there?

ff_goal = abs(cc(1:end-1,end));
a0 = sqrt(s.^2.*(1-ff_goal)./ff_goal);

%%

[a_local2,fval,exitflag,output] = fminunc(@(x)ridge_mse(x,xs,ys,xsT,zscore1(y_test,y_m,y_std)),a0,options)

B_opt3 = (xs'*xs+diag(a_local2.^2))\(xs'*ys);
yp_opt3 = xsV*B_opt2;
yp_opt3 = unscore(yp_opt3,y_m,y_std);
RMSE_opt3 = sqrt(mean((yp_opt3-y_val).^2))

figure(f)
plot(yp_opt3,'y')
legend('Actual','CV-method','Linear Regression','Ridge Reg - opt','Local Ridge Reg','Local Ridge 2','location','best')

RMSE_linReg
RMSE_CV
RMSE_opt
RMSE_opt2
RMSE_opt3

%%
figure; bar(abs([a_local a_local2])); legend('Original','Seeded')

ff_local2 = s.^2./(s.^2 + a_local2.^2);

figure;
plot(abs(cc(end,1:end-1)),'o-')
hold on
plot(ff_local2,'s-')
legend('Correlation to output','Local Filter Factor','location','best')

%%

figure(f)
legend('Actual','CV-method','Linear Regression','Ridge Reg - opt','Local Ridge Reg','Local Ridge 2','Local Ridge 3','location','best')

RMSE_linReg
RMSE_CV
RMSE_opt
RMSE_opt2

figure;
plot(abs(cc(end,1:end-1)),'o-')
hold on
plot(ff_local3,'s-')
legend('Correlation to output','Local Filter Factor','location','best')

figure; bar(abs([a_local a_local2 a_local3])); legend('Original','Seeded','MultiStart')

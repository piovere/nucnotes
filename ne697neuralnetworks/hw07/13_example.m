%%%---------------------------------------------------------------------%%%
%%%                                                                     %%%
%%%                           Ridge Regression                          %%%
%%%                                                                     %%%
%%%---------------------------------------------------------------------%%%


clear
clc
close all

% Load in the car data %
% Variables in data are:
% 1. Number of Cylinders
% 2. Displacement
% 3. Horsepower
% 4. Weight
% 5. Acceleration (time to go from 0-60 mph in seconds)
% 6. Year
% 7. MPG

load ExampleData
whos

[numCars numVars] = size(data);

%% This is all copied from the previous examples! %%

y = data(:,5);
x = data(:,[1:4 6 7]);
% Add in inverse horsepower and weight/horsepower, since we know they have
% a linear relationship with acceleration time %
x = [x 1./data(:,3) data(:,4)./data(:,3)];

% We can divide the data into training, test, and validation sets using 
% even/odd division because there is no relationship between adjacent
% observations

train_ind = 2:2:numCars; % even observations
test_ind = 1:4:numCars; % every other odd observation (1, 5, 9, ..)
val_ind = 3:4:numCars; % every other odd observation (3, 7, 11, ..)

x_train = x(train_ind,:);
y_train = y(train_ind,:);
x_test = x(test_ind,:);
y_test = y(test_ind,:);
x_val = x(val_ind,:);
y_val = y(val_ind,:);

% Check to see if the training data includes the min/max values. If not - add them in!

% Identify variables where min value is not in x_train and add appropriate
% observations
indObs = [];

indVar = find(min(x)<min(x_train));
for ii = 1:numel(indVar)
    indObs(ii) = find(x(:,indVar(ii)) == min(x(:,indVar(ii))),1);
end
x_train = [x_train;x(unique(indObs),:)];
y_train = [y_train;y(unique(indObs))];

% Identify variables where max value is not in x_train and add appropriate
% observations
indObs = [];

indVar = find(max(x)>max(x_train));
for ii = 1:numel(indVar)
    indObs(ii) = find(x(:,indVar(ii)) == max(x(:,indVar(ii))),1);
end
x_train = [x_train;x(unique(indObs),:)];
y_train = [y_train;y(unique(indObs))];


%% 

% For Ridge Regression, we want to standardize the inputs AND output %
[xs x_m x_std] = zscore1(x_train);
[ys y_m y_std] = zscore1(y_train);
xsT = zscore1(x_test,x_m,x_std); % standardize the test inputs using the same mean/std dev!
xsV = zscore1(x_val,x_m,x_std); % standardize validation inputs using the same mean/std dev!

%% Let's look at how the condition number has changed just with
% standardization

initial_cond = cond(x_train'*x_train)
standardized_cond = cond(xs'*xs)

% to look at it another way .. 
eigs = eig(x_train'*x_train)
max(eigs)/min(eigs)
eigs = eig(xs'*xs)
max(eigs)/min(eigs)

s = svd(xs)  % singular values of the standardized data

%% We want to test regularization parameters that cover the range of
% singular values, so we'll use alphas in the range of [0.1 100] (for
% convenience)

% First, we'll do an L-curve method to find our optimal alpha

alpha = logspace(-1,2,100);
RMSE = nan(size(alpha)); % Mean squared error for each alpha
normB = nan(size(alpha)); % size the regression coefficients for each alpha
cond_num = nan(size(alpha));

for ii = 1:numel(alpha)
    B = ridge(ys,xs,alpha(ii).^2);
    normB(ii) = norm(B);
    yps = xs*B; % predictions for training data %
    yps = unscore(yps,y_m,y_std); % un-standardize predictions %
    error = y_train - yps;
    RMSE(ii) = sqrt(mean(error.^2));
    e = eig(xs'*xs + alpha(ii).^2*eye(size(xs,2)));
    cond_num(ii) = max(e)/min(e);
end

figure; semilogx(alpha,RMSE)
xlabel('\alpha')
ylabel('RMSE')

figure; semilogx(alpha,normB)
xlabel('\alpha')
ylabel('|B|')

figure; plot(RMSE,normB,'o-')
title('Solution Norm vs Mean Squared Error')
xlabel('RMSE')
ylabel('|B|')

figure; semilogx(alpha,cond_num);
xlabel('\alpha')
ylabel('Condition Number')

%%

% Best solution appears to be around normB = 0.55 %
optnormB = 0.55;

ind = find(min(abs(normB - optnormB)) == abs(normB - optnormB));
[min(RMSE) RMSE(ind)]
a = alpha(ind)
cond_from_s = (max(s)^2 + a^2)/(min(s)^2 + a^2)
con_L = cond_num(ind)


%% Now, we'll do a cross-validation method
% we'll expand the alpha space to be [0]U[0.1,100]

alpha = [0 logspace(-1,2,100)];
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

figure; semilogx(alpha,RMSE_train)
hold all
semilogx(alpha,RMSE_test)
legend('Training','Test','location','best')
xlabel('\alpha')
ylabel('RMSE (s)')
title('RMSE of Test Data')

ind = find(RMSE_test == min(RMSE_test));
a_CV = alpha(ind)
RMSE_min = RMSE_test(ind)
con_min = cond_num(ind)
cond_from_s = (max(s)^2 + a_CV^2)/(min(s)^2 + a_CV^2)
RMSE_lin = RMSE_test(1) % RMSE for OLS regression - alpha = 0

a_CV2 = 2.009; % based on visual inspection 
%% Comparing the two methods on test data

% for L-curve method
BL = ridge(ys,xs,a^2);
ypL = xsT*BL;
ypL = unscore(ypL,y_m,y_std);
RMSE_L = sqrt(mean((ypL-y_test).^2))

% for CV method (absolute minimum of test error)
BCV = ridge(ys,xs,a_CV^2);
ypCV = xsT*BCV;
ypCV = unscore(ypCV,y_m,y_std);
RMSE_CV = sqrt(mean((ypCV-y_test).^2))

% for CV2 method (visual inspection of test error)
BCV2 = ridge(ys,xs,a_CV2^2);
ypCV2 = xsT*BCV2;
ypCV2 = unscore(ypCV2,y_m,y_std);
RMSE_CV2 = sqrt(mean((ypCV2-y_test).^2))

% for regular linear regression
BLR = regress(ys,xs);
ypLR = xsT*BLR;
ypLR = unscore(ypLR,y_m,y_std);
RMSE_linReg = sqrt(mean((ypLR-y_test).^2))

figure; 
plot(y_test,'k.')
hold all
plot(ypL,'b')
plot(ypCV,'r-s')
plot(ypCV2,'m-o')
plot(ypLR,'g')
legend('Actual','L-method','CV-method','CV2-method','Linear Regression','location','best')

%% RESULTS SO FAR

% Predicting Acceleration Time from other variables (test RMSE)

% Regression:
% Full model - 1.7309 s
% Correlation model - 1.7485 s
% Single variable - 2.0086 s
% Two variables - 1.7689 s
% Three variables - 1.7094 s
% Quadratic (no interaction) - 1.4657 s
% Reduced quadratic - 1.4926 s
% Inverse HP - 1.5087 s
% Force term - 1.4571 s
% Quadratic and Force - 1.4330 s
% Reduced full model - 1.4321 s
% Random nonlinear model - 1.5711 s
%
% PCR:
% 95% Var exp - 2.1823 s
% PC 1 - 2.2174 s
% PC 1,2 - 2.2211 s
% PC 1,3,4 - 2.0289 s
% PCs 1,3,4,5 - 1.7180s
% All PCs - 1.7309 s
%
% PLS:
% 1 LV - 2.1917 s
% 2 LV - 1.9800 s
% 3 LV - 1.8797 s
% 4 LV - 1.7534 s
% 5 LV - 1.7314 s
% 6 LV - 1.7309 s
% with nonlinear term ..
% 1 LV - 1.9291 s
% 2 LV - 1.5091 s
% 3 LV - 1.5331 s
% 4 LV - 1.5114 s
% 5 LV - 1.4393 s
% 6 LV - 1.4530 s
% 7 LV - 1.4571 s
%
% Ridge Regression:
% L-curve (alpha = 5.0941) - 1.5069 s
% CV (alpha = 0.3430) - 1.4496 s
% CV2 (alpha = 1.262) - 1.4529 s

%% %% finally, evaluate the best model on the validation data ..
% best model was using cross-validation (alpha = 0.3511)

% build model just to be sure we have the right one %
BCV = ridge(ys,xs,a_CV^2);
ypCV = xsV*BCV;
ypCV = unscore(ypCV,y_m,y_std);
RMSE_val = sqrt(mean((ypCV-y_val).^2))

%% Validation results for all BEST models

% Regression: 1.4535 s
% PCR: 1.7648 s
% PLS: 1.4858 s
% Ridge Regression: 1.4554


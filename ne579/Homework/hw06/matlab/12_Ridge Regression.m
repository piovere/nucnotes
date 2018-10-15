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

varNames = {'NumCyl','Disp','HP','Weight','AccTime','Year'};

%% Here we want to use some subset of variables 1-6 to predict MPG (var 7) %%


[numCars numVar] = size(data); 
numVar = numVar-1; % number of inputs - one variable is the output!

% We can divide the data into training, test, and validation sets using 
% even/odd division because there is no relationship between adjacent
% observations

train_ind = 2:2:numCars; % even observations
test_ind = 1:4:numCars; % every other odd observation (1, 5, 9, ..)
val_ind = 3:4:numCars; % every other odd observation (3, 7, 11, ..)

train = data(train_ind,:);
test = data(test_ind,:);
val = data(val_ind,:);

% Check to see if the training data includes the min/max values. If not - add them in!

indObs = [];

% check minimums:
[min(data)' min(train)']
% Identify variables where min value is not in x_train and add appropriate
% observations
indVar = find(min(data)<min(train))
for ii = 1:numel(indVar)
    indObs(ii) = find(data(:,indVar(ii)) == min(data(:,indVar(ii))),1);
end 

train = [train;data(unique(indObs),:)];

indObs = [];

% check maximums:
[max(data)' max(train)']
% Identify variables where max value is not in x_train and add appropriate
% observations
indVar = find(max(data)>max(train))
for ii = 1:numel(indVar)
    indObs(ii) = find(data(:,indVar(ii)) == max(data(:,indVar(ii))),1);
end
train = [train;data(unique(indObs),:)];

% final check
[min(data)' min(train)' max(data)' max(train)']

% divide inputs from outputs

x_train = train(:,1:6);
y_train = train(:,7);
x_test = test(:,1:6);
y_test = test(:,7);
x_val = val(:,1:6);
y_val = val(:,7);


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

%% We want to test regularization parameters that cover the range of
% singular values, so we'll use alphas in the range of [0.1 100] (for
% convenience)

s = svd(xs)  % singular values of the standardized data

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

% Best solution appears to be around normB = 0.35 %
optnormB = 0.35;

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

% Predicting MPG from other variables (test RMSE)

% Regression:
% Full model - 3.77 MPG
% Correlation model - 4.66 MPG
% Single variable - 4.72 MPG
% Two variables - 3.73 MPG
% Three variables - 3.71 MPG
% Quadratic (no interaction) - 2.86 MPG
% Reduced quadratic - 3.01 MPG
% Inverse Weight - 3.31 MPG
% Quadratic and Inverse Weight - 2.86 MPG
% Reduced full model - 3.05 MPG
% Random nonlinear model - 3.69 MPG
%
% PCR:
% 95% Var exp - 3.95 MPG
% PC 1 - 4.34 MPG
% PC 1,2 - 4.08 MPG
% PC 1,3 - 4.22 MPG
% All PCs - 3.77 MPG
%
% PLS:
% 1 LV - 4.17 MPG
% 2 LV - 3.90 MPG
% 3 LV - 3.95 MPG
% 4 LV - 3.85 MPG
% 5 LV - 3.76 MPG
% 6 LV - 3.77 MPG <- This is the same as the 6-term OLS and the 6 PC PCR models!
% with quadratic terms ..
% 1 LV - 4.33 MPG
% 2 LV - 4.08 MPG
% 3 LV - 3.92 MPG
% 4 LV - 3.53 MPG
% 5 LV - 3.23 MPG
% 6 LV - 3.02 MPG
% 7 LV - 3.02 MPG
% 8 LV - 3.01 MPG
% 9 LV - 2.94 MPG
% 10 LV - 2.84 MPG
% 11 LV - 2.86 MPG
% 12 LV - 2.86 MPG <- This is the same as the quadratic linear regression model! 
%
% Ridge Regression
% L-curve (alpha = 15.1991) - 4.49 MPG
% Cross validation 1 (alpha = 0) - 3.77 MPG <- this is just linear regression!
% Cross validation 2 (alpha = 2.009) - 3.80 MPG

%% finally, evaluate the best model on the validation data ..
% best model was using cross-validation 

% build model just to be sure we have the right one %
BCV = ridge(ys,xs,a_CV^2);
ypCV = xsV*BCV;
ypCV = unscore(ypCV,y_m,y_std);
RMSE_val = sqrt(mean((ypCV-y_val).^2))

%% Validation results for all BEST models

% Regression: 2.93 MPG
% PCR: 3.52 MPG (note this has no nonlinear terms!)
% PLS: 2.88 MPG
% Ridge Regression: 3.31 MPG (also no nonlinear terms!)


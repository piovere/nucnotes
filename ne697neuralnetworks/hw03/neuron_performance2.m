% Ask the user to input a data file of training data
fn = input('Input the filename with training data: ', 's');
% maybe verify to see if the file contained an "x" and a "y"
% verify it contains exactly two matrices
% verify that they have the same first dimension
% verify their names are "x" and "y"
% load the data
load(fn)

% MCUV scales the input data (call zscore1)
[xs_mcuv, std_xs, mean_xs] = zscore1(x);
[ys_mcuv, std_ys, mean_ys] = zscore1(y);

%n_array = [];
%p_array = [];

net2 = feedforwardnet(1);
net2 = init(net2);
net2.trainParam.goal = 1;
net2.performFcn = 'sse';
net2.divideFcn = 'dividetrain';
net2.layers{1}.transferFcn = 'logsig';
[net2, tr] = train(net2, xs_mcuv', ys_mcuv');
%n_array = [n_array i];
%p_array = [p_array tr.best_perf];
clear net2;

plt = semilogx(n_array, p_array);
title('Performance vs. Number of Neurons');
xlabel('Neurons');
ylabel('SSE');

saveas(plt, 'r
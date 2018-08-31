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

% Trains the network using the trainlm method (this is done with the newff
% and train functions)
net = feedforwardnet(5);
net = init(net);
net.trainParam.goal = 1e-10;
net.performFcn = 'sse';
net.divideFcn = 'dividetrain';
net.layers{1}.transferFcn = 'logsig';
[net, tr] = train(net, xs_mcuv', ys_mcuv');

% If training is successful (SSE < 1), saves the weight matrices in a file.
if tr.best_tperf < 1
    save('trainednetwork', 'net');
end
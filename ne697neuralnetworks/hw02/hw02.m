Error = [];
W = [];
B = [];

lrs = [0.1, 0.3, 0.5, 0.7, 0.99]
lr = 0.1;

rand('seed',10);
x = rand(100, 2);

Y = 2 * x(:,1) - 3 * x(:,2) - 1;

[net] = newlin(x', Y', [0], lr);

for i = 1:100
    y = 2 * x(:,1) - 3 * x(:,2) - 1 + 0.02 * i;
    [net, Y, E] = adapt(net, x(i,:)', y(i));
    w = net.IW{1};
    b = net.b{1};
    W = [W;w];
    B = [B;b];
    Error = [Error E];
end

x_axis = 1:100;

figure();
plot(Error);
title(join(['Error for learning rate',string(lr)]));
ylabel('Error');
xlabel('Epoch');
legend('Error');
drawnow

figure();
plot(x_axis, W', x_axis, B');
legend('x\_1 weight', 'x\_2 weight', 'Bias');
title(join(['Weights and Bias vs. Training Epoch (learning rate =',string(lr),')']));
xlabel('Epoch');
drawnow
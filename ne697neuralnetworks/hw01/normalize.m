filename = input('Enter a filename: ','s');
load(filename);
% have to figure out what the name of the matrix we just loaded is
vs = whos('-file',filename);
% store the loaded variable in a name that I know
m = eval(vs.name);
[x_mcuv, std_x, mean_x] = zscore1(m)
save('parameters','x_mcuv','std_x','mean_x');
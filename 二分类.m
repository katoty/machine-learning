clear all;clc;

%data

X = [2 1 ;2 2;5 4;4 5;2 3; 3 2 ;6 5;4 1;6 3;7 4];
Y = [0;0;1;1;0;0;1;1;0;0];

%参数

[m,n] = size(x);
X_enlarge = [X ones(m,1)];
belta1 = randn(n+1,1);
alpha = 0.001;
internum = 100;

for i = 1:internum
    Z = X_enlarge*belta1;
    P = 1./(1+exp(-Z));
    E = Y-P;
    belta2 = belta1-alpha*X_enlarge'*E;
    belta1 = belta2;
end

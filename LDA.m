x0 = randi([1, 10], [1, 10])';
y0 = randi([0, 1], [1, 10])';
X0 = [x0,y0];
scatter(x0, y0,'r','filled');
hold on;

x1 = randi([1, 10], [1, 10])';
y1 = randi([0, 1], [1, 10])';
X1 = [x1,y1];
scatter(x1, y1,'y','filled');

% 均值向量
u0 = mean(X0)';
u1 = mean(X1)';

% 类内散度矩阵
Sw = cov(X0) + cov (X1);

% %类间散度矩阵
% Sb = (u0-u1)*(u0-u1)';

w = Sw \ (u0 - u1);

x = linspace(0, 10, 100);
y = (-w(1) * x) / w(2); 

% Plot
plot(x, y);
hold off;

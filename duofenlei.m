
clear;clc;
data = load("Iris.mat");
Class = data.Class;
Feature = zscore(data.Feature);

%% train

x_train = Feature([1:30,51:80,101:130],:);

y_train = Class([1:30,51:80,101:130],:);


x_c1 = x_train(1:30,:);
x_c2 = x_train(31:60,:);
x_c3 = x_train(61:90,:);

y_c1 = y_train(1:30,:);
y_c2 = y_train(31:60,:);
y_c3 = y_train(61:90,:);

%% test

x_t1 = Feature(31:50,:);
x_t2 = Feature(81:100,:);
x_t3 = Feature(131:150,:);
x_test =[x_t1;x_t2;x_t3];

y_t1 = Class(31:50,:);
y_t2 = Class(81:100,:);
y_t3 = Class(131:150,:);
y_test = [y_t1;y_t2;y_t3];

%% LDA

sigma1 = 29*cov(x_c1);
sigma2 = 29*cov(x_c2);
sigma3 = 29*cov(x_c3);

sw = sigma1+sigma2+sigma3;
X = [x_c1; x_c2; x_c3];
st = 89* cov(X);
sb = st - sw;


%% V, 一个矩阵，其列是输入矩阵的特征向量。
%% D, 一个对角矩阵，其对角线上的元素是输入矩阵的特征值
[V,D] = eig (sw\sb);

%% 降维并绘图 （w为4*2大小的投影矩阵将属性数为4的样本的投影到2维空间）
w = V(:,1:2);

f = @(x)x*w;

y1 = f(x_c1);
scatter(y1(:,1),y1(:,2),'+')
hold on;
y2 = f(x_c2);
scatter(y2(:,1),y2(:,2),'^')
y3 = f(x_c3);
scatter(y3(:,1),y3(:,2),'x')


hold off;
legend('第一类', '第二类', '第三类');
title("LDA降维后数据");

%% 利用 lda 中函数进行分类，并给出待预测数据在分类器中类别
[m,~] = size(x_test);
arr=[];
for i=1:m
    % c1 为 +  c2为 -
    class1 = LDA(x_test(i,:)*w,x_c1*w,x_c2*w);
    % c1 为 +  c3为 -
    class2 = LDA(x_test(i,:)*w,x_c1*w,x_c3*w);
    % c2 为 +  c3为 -
    class3 = LDA(x_test(i,:)*w,x_c2*w,x_c3*w);
    
    %% 计算票数
    c1 = 1 * (class1) + 1 * (class2);
    c2 = 1 * (~class1) + 1 * (class3);
    c3 = 1 * (~class2) + 1 * (~class3);

    c= [c1 c2 c3];

    [max_value, max_index] = max(c);
    arr(i,1) = max_index;
end

%%  为直观起见，将 1 0 0  ； 0 1 0；0 0 1；记为第1 2 3类
disp("LDA预测结果")
disp(arr)
disp("正确率")
disp(1-err(arr))


%% 对数几率回归 
%% OVO拆分策略 训练三个分类器

%% 利用logit.m内的函数训练分类器 并给出待预测数据在分类器中类别
[m,~] = size(x_test);
arr1 = zeros(m,1);
for i=1:m
    % c1 为 +  c2为 -
    class1 = logit(x_test(i,:),x_c1,y_c1,x_c2,y_c2);
    % c1 为 +  c3为 -
    class2 = logit(x_test(i,:),x_c1,y_c1,x_c3,y_c3);
    % c2 为 +  c3为 -
    class3 = logit(x_test(i,:),x_c2,y_c2,x_c3,y_c3);
    
    %% 计算票数
    c1 = 1 * (class1) + 1 * (class2);
    c2 = 1 * (~class1) + 1 * (class3);
    c3 = 1 * (~class2) + 1 * (~class3);

    c= [c1 c2 c3];

    [max_value, max_index] = max(c);
    arr1(i,1) = max_index;
end
disp("对数几率回归预测");
disp(arr1);
disp("正确率");
disp(1-err(arr1));


















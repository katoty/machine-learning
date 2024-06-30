clear;clc;close all;
load('ORL_32x32.mat', 'fea', 'gnd');
X = fea;
y = gnd;

% 设置随机数种子以便复现结果
rng(42);

% 初始化
num_samples = size(X, 1);
num_classes = 40; % ORL数据集有40个人
samples_per_class = 10; % 每个类别的样本数
train_samples_per_class = 6; % 每个类别训练样本数
val_samples_per_class = 2; % 每个类别验证样本数
test_samples_per_class = samples_per_class - train_samples_per_class - val_samples_per_class; % 每个类别测试样本数

% 初始化索引
train_idx = [];
val_idx = [];
test_idx = [];

for i = 1:num_classes
    % 找到每个类别的样本索引
    class_idx = find(y == i);
    
    % 随机打乱每个类别的样本索引
    class_idx = class_idx(randperm(length(class_idx)));
    
    % 将每个类别的样本分为训练集、验证集和测试集
    train_idx = [train_idx; class_idx(1:train_samples_per_class)];
    val_idx = [val_idx; class_idx(train_samples_per_class + 1:train_samples_per_class + val_samples_per_class)];
    test_idx = [test_idx; class_idx(train_samples_per_class + val_samples_per_class + 1:end)];
end

% 划分训练集、验证集和测试集
X_train = X(train_idx, :);
y_train = y(train_idx);
X_val = X(val_idx, :);
y_val = y(val_idx);
X_test = X(test_idx, :);
y_test = y(test_idx);

% 进行PCA
[coeff, X_train_pca, ~, ~, explained] = pca(X_train);

% 选择保留的主成分数目（例如，解释方差达到95%的主成分）
num_components = find(cumsum(explained) >= 95, 1);

% 将训练集、验证集和测试集投影到PCA空间
X_train_pca = X_train * coeff(:, 1:num_components);
X_val_pca = X_val * coeff(:, 1:num_components);
X_test_pca = X_test * coeff(:, 1:num_components);

% 训练多分类SVM分类器
%SVMModel = fitcecoc(X_train_pca, y_train);

% 训练多分类SVM分类器
t = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', 1, 'KernelScale', 'auto');
% one vs one
% SVMModel = fitcecoc(X_train_pca, y_train, 'Learners', t, 'Coding', 'onevsone');
% one vs others
SVMModel = fitcecoc(X_train_pca, y_train, 'Learners', t, 'Coding', 'onevsall');


% 使用验证集进行预测以调整模型参数
y_val_pred = predict(SVMModel, X_val_pca);

% 计算验证集准确率
val_accuracy = sum(y_val_pred == y_val) / length(y_val);
fprintf('验证集准确率: %.2f%%\n', val_accuracy * 100);

% 使用训练好的SVM模型进行测试集预测
y_test_pred = predict(SVMModel, X_test_pca);

% 计算测试集准确率
test_accuracy = sum(y_test_pred == y_test) / length(y_test);
fprintf('测试集准确率: %.2f%%\n', test_accuracy * 100);

% 计算混淆矩阵
confMat = confusionmat(y_test, y_test_pred);

% 可视化混淆矩阵
figure;
confusionchart(confMat);
title('混淆矩阵');


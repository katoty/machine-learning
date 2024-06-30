clear; clc;close all;
load('glass.mat'); % 存放眼镜labels标签
load('ORL_32x32.mat', 'fea', 'gnd');

x = fea;
y = labels;

% 划分比例
trainRatio = 0.7;
valRatio = 0.15;
testRatio = 0.15;

% 设置随机种子方便比较超参数改动对模型的影响
rng(1); % 设置随机种子

% 划分数据
[trainInd, valInd, testInd] = dividerand(400, trainRatio, valRatio, testRatio);

% 划分输入数据和标签
trainData = x(trainInd, :);
valData = x(valInd, :);
testData = x(testInd, :);

trainLabels = y(trainInd);
valLabels = y(valInd);
testLabels = y(testInd);

% 输出样本数量
numTrainSamples = length(trainInd);
numValSamples = length(valInd);
numTestSamples = length(testInd);

disp(['训练集样本数: ', num2str(numTrainSamples)]);
disp(['验证集样本数: ', num2str(numValSamples)]);
disp(['测试集样本数: ', num2str(numTestSamples)]);

%% 对比线性核高斯核效果，高斯核更好
% 训练SVM模型
% 使用线性核函数
%SVMModel = fitcsvm(trainData, trainLabels);
% 使用 高斯核函数
SVMModel = fitcsvm(trainData, trainLabels, 'KernelFunction', 'rbf', 'KernelScale', 'auto');


% 验证模型
valPredictions = predict(SVMModel, valData);
valAccuracy = sum(valPredictions == valLabels) / numel(valLabels);
disp(['验证集准确率: ', num2str(valAccuracy)]);

% 测试模型
testPredictions = predict(SVMModel, testData);
testAccuracy = sum(testPredictions == testLabels) / numel(testLabels);
disp(['测试集准确率: ', num2str(testAccuracy)]);

% 计算混淆矩阵
testConfMat = confusionmat(testLabels, testPredictions);
disp('混淆矩阵 (测试集):');
disp(testConfMat);

% 计算查准率、查全率和F1值
numClasses = length(unique(testLabels));
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1 = zeros(numClasses, 1);

for i = 1:numClasses
    tp = testConfMat(i,i);
    fp = sum(testConfMat(:,i)) - tp;
    fn = sum(testConfMat(i,:)) - tp;
    
    precision(i) = tp / (tp + fp);
    recall(i) = tp / (tp + fn);
    f1(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

% 输出查准率、查全率和F1值表格
classLabels = unique(testLabels);
metricsTable = table(classLabels, precision, recall, f1);
disp('查准率、查全率和F1值表:');
disp(metricsTable);

% 可视化混淆矩阵
figure;
confusionchart(testLabels, testPredictions);
title('混淆矩阵 (测试集)');


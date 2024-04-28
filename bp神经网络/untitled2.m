clear;
data = load("Iris.mat");
feature_data = data.Feature;
Class_data = data.Class;

% 训练集

training_feature = feature_data([1:30,51:80,101:130],:);
training_Class = Class_data([1:30,51:80,101:130],:);

% 测试集
test_feature = feature_data([31:50,81:100,131:150],:);
test_Class = Class_data([31:50,81:100,131:150],:);


% 按列标准化
training_feature_normalized = zscore(training_feature);
test_feature_normalized = zscore(test_feature);


% 假设隐层神经元个数为num_hidden_neurons 
num_input_neurons = size(training_feature_normalized, 2);
num_hidden_neurons = 8;
num_output_neurons = size(training_Class, 2);

% 创建并配置神经网络
net = feedforwardnet(num_hidden_neurons);

% 设置学习速率和迭代次数
net.trainParam.lr = 0.001;  
net.trainParam.epochs = 1000; 

% 使用误差逆传播算法训练神经网络
net = train(net, training_feature_normalized', training_Class');


accuracy = 0;
num =5;
% 多次计算得到平均精度
for i = 1:num
    % 计算训练集的预测结果
    predicted_output = net(test_feature_normalized')';
    
    
    % 找到每个样本中概率最大的类别的索引
    [~, predicted_class_index] = max(predicted_output, [], 2);
    [~, true_class_index] = max(test_Class, [], 2);
    
    % 比较预测结果和实际类别数据
    correct_predictions = (predicted_class_index == true_class_index);
    
    % 计算准确率
    accuracy = accuracy +sum(correct_predictions) / size(test_Class, 1);
    
end
disp(['学习速率为',num2str(net.trainParam.lr),'隐层神经元个数为',num2str(num_hidden_neurons),'测试集准确率为: ', num2str(accuracy/num)]);

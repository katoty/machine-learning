load mnist_uint8;

train_x = double(reshape(train_x', 28, 28, 60000)) / 255;
test_x = double(reshape(test_x', 28, 28, 10000)) / 255;
train_y = double(train_y');
test_y = double(test_y');

opts.batchsize = 50;
opts.numepochs = 1;

% 定义不同的学习率
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5, 1];
test_errors = zeros(length(learning_rates), 1); % 存储每种配置下的测试错误率

for i = 1:length(learning_rates)
    learning_rate = learning_rates(i);
    opts.alpha = learning_rate;
    
    rand('state', 0)
    cnn.layers = {
        struct('type', 'i') % 输入层
        struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) % 第一个卷积层
        struct('type', 's', 'scale', 2) % 第一个子采样层
        struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) % 第二个卷积层
        struct('type', 's', 'scale', 2) % 第二个子采样层
    };
    cnn = cnnsetup(cnn, train_x, train_y);
    
    cnn = cnntrain(cnn, train_x, train_y, opts);
    
    [er, ~] = cnntest(cnn, test_x, test_y);
    test_errors(i) = er;
end

% 绘制学习率与测试错误率的关系图
figure;
plot(learning_rates, test_errors, '-o');
xlabel('Learning Rate');
ylabel('Test Error Rate');
title('Test Error Rate vs Learning Rate');
grid on;

disp('Test Errors for different learning rates:');
disp(table(learning_rates', test_errors, 'VariableNames', {'Learning Rate', 'Test Error Rate'}));

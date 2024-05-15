load mnist_uint8;

train_x = double(reshape(train_x', 28, 28, 60000)) / 255;
test_x = double(reshape(test_x', 28, 28, 10000)) / 255;
train_y = double(train_y');
test_y = double(test_y');

opts.alpha = 1;
opts.batchsize = 50;

% 定义不同的训练轮数
num_epochs = [1, 3, 5, 7];
test_errors = zeros(length(num_epochs), 1); % 存储每种配置下的测试错误率

for i = 1:length(num_epochs)
    opts.numepochs = num_epochs(i);
    
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

% 绘制训练轮数与测试错误率的关系图
figure;
plot(num_epochs, test_errors, '-o');
xlabel('Number of Epochs');
ylabel('Test Error Rate');
title('Test Error Rate vs Number of Epochs');
grid on;

disp('Test Errors for different number of epochs:');
disp(table(num_epochs', test_errors, 'VariableNames', {'Number of Epochs', 'Test Error Rate'}));

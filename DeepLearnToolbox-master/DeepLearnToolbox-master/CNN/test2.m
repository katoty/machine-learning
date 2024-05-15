load mnist_uint8;

train_x = double(reshape(train_x', 28, 28, 60000)) / 255;
test_x = double(reshape(test_x', 28, 28, 10000)) / 255;
train_y = double(train_y');
test_y = double(test_y');

opts.alpha = 1;
opts.numepochs = 1;

% 定义不同的批量大小
batch_sizes = [10, 20, 50, 100, 200, 500];
test_errors = zeros(length(batch_sizes), 1); % 存储每种配置下的测试错误率

for i = 1:length(batch_sizes)
    batch_size = batch_sizes(i);
    opts.batchsize = batch_size;
    
    rand('state', 0)
    cnn.layers = {
        struct('type', 'i') % 输入层
        struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) % 第一个卷积层
        struct('type', 's', 'scale', 2) % 第一个子采样层
        struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) % 第二个卷积层
        struct('type', 's', 'scale', 2) % 第二个子采采层
    };
    cnn = cnnsetup(cnn, train_x, train_y);
    
    cnn = cnntrain(cnn, train_x, train_y, opts);
    
    [er, ~] = cnntest(cnn, test_x, test_y);
    test_errors(i) = er;
end

% 绘制批量大小与测试错误率的关系图
figure;
plot(batch_sizes, test_errors, '-o');
xlabel('Batch Size');
ylabel('Test Error Rate');
title('Test Error Rate vs Batch Size');
grid on;

disp('Test Errors for different batch sizes:');
disp(table(batch_sizes', test_errors, 'VariableNames', {'Batch Size', 'Test Error Rate'}));

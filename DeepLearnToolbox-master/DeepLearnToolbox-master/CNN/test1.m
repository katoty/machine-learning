load mnist_uint8;

train_x = double(reshape(train_x', 28, 28, 60000)) / 255;
test_x = double(reshape(test_x', 28, 28, 10000)) / 255;
train_y = double(train_y');
test_y = double(test_y');

opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 1;

% 定义不同的卷积核个数配置，使用元组（第一个卷积层卷积核个数，第二个卷积层卷积核个数）
conv_kernels = [4, 8; 6, 12; 8, 16; 10, 20; 12, 24; 14, 28; 16, 32];
test_errors = zeros(size(conv_kernels, 1), 1); % 存储每种配置下的测试错误率

for i = 1:size(conv_kernels, 1)
    num_kernels_1 = conv_kernels(i, 1);
    num_kernels_2 = conv_kernels(i, 2);
    
    rand('state', 0)
    cnn.layers = {
        struct('type', 'i') % 输入层
        struct('type', 'c', 'outputmaps', num_kernels_1, 'kernelsize', 5) % 第一个卷积层
        struct('type', 's', 'scale', 2) % 第一个子采样层
        struct('type', 'c', 'outputmaps', num_kernels_2, 'kernelsize', 5) % 第二个卷积层
        struct('type', 's', 'scale', 2) % 第二个子采样层
    };
    cnn = cnnsetup(cnn, train_x, train_y);
    
    cnn = cnntrain(cnn, train_x, train_y, opts);
    
    [er, ~] = cnntest(cnn, test_x, test_y);
    test_errors(i) = er;
end

% 绘制卷积核个数与测试错误率的关系图
figure;
plot(conv_kernels(:, 1), test_errors, '-o');
xlabel('Number of Kernels in First Conv Layer');
ylabel('Test Error Rate');
title('Test Error Rate vs Number of Kernels in First Conv Layer');
grid on;

disp('Test Errors for different number of kernels:');
disp(table(conv_kernels(:, 1), conv_kernels(:, 2), test_errors, 'VariableNames', {'Conv1 Kernels', 'Conv2 Kernels', 'Test Error Rate'}));

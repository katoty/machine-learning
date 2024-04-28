% 读取数据集
data = {...
    '青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是';...
    '乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是';...
    '乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是';...
    '青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是';...
    '浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是';...
    '青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是';...
    '乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是';...
    '乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是';...
    '乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否';...
    '青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否';...
    '浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否';...
    '浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否';...
    '青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否';...
    '浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否';...
    '乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否';...
    '浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否';...
    '青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否'...
};

% 提取特征和标签
X_train = data([1, 2, 3, 6, 7, 10, 14, 15, 16, 17], 1:6); % 训练集特征
y_train = categorical(data([1, 2, 3, 6, 7, 10, 14, 15, 16, 17], end)); % 训练集标签
X_valid = data([4, 5, 8, 9, 11, 12, 13], 1:6); % 验证集特征
y_valid = categorical(data([4, 5, 8, 9, 11, 12, 13], end)); % 验证集标签

% 使用独热编码转换特征数据
X_train_encoded = [];
X_valid_encoded = [];

for i = 1:size(X_train, 2)
    % 获取当前特征列的唯一值
    unique_values = unique(X_train(:, i));
    
    % 为每个唯一值创建一个新列
    for j = 1:length(unique_values)
        % 判断当前特征列中的值是否等于当前唯一值
        encoded_column = strcmp(X_train(:, i), unique_values{j});
        X_train_encoded = [X_train_encoded encoded_column];
        
        % 对验证集做同样的处理
        encoded_column_valid = strcmp(X_valid(:, i), unique_values{j});
        X_valid_encoded = [X_valid_encoded encoded_column_valid];
    end
end

% 构建决策树模型
tree = fitctree(X_train_encoded, y_train, 'prune', 'on');

% 可视化决策树
view(tree, 'Mode', 'graph');

% 在验证集上进行预测
y_pred = predict(tree, X_valid_encoded);

% 计算准确率
accuracy = sum(strcmp(y_pred, y_valid)) / numel(y_valid);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);


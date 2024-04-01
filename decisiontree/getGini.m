%***************************************
%  featureIndex 表示要计算的基尼指数的特征在数据集中的列索引。
%****************************************

function gini_Index = getGini(data, featureIndex)

    % 转化数据为矩阵类型
    data = table2array(data);
 
    % 获取数据集中的所有类别    
    % |y|
    classes = unique(data(:,  end));


    % 获取特定特征的所有可能取值
    % 其元素个数即为V
    featureValues = unique(data(:, featureIndex));

    % 初始化属性的gini指数
    gini_Index = 0;

    % 获取数据的总样本数
    % |D|
    totalSamples = size(data, 1);

    for i = 1:numel(featureValues)

            % % 获取当前取值  a_i 的数据
            featureData = data(data(:, featureIndex) == featureValues(i), :);

            % 获取当前取值的样本数 D^i
            featureSize = size(featureData, 1);

            %计算第i类样本比例  D^i/D
            ratio_i = featureSize / totalSamples;

            %% 计算gini(Dv)
            p = 0;
            for k =1 :numel(classes)
                 % 统计第 k 类的样本数量
                classCount = sum(featureData(:, end) == classes(k));
    
                % 计算第 k 类在数据集中的比例

                p = p + (classCount /featureSize )*(classCount /featureSize );
            end


            gini_D_v = 1 - p;
         
        gini_Index = gini_Index + ratio_i * gini_D_v;
    end
    end



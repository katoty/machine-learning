%****************************************
%getGain.m
%****************************************
%计算信息增益
function [gain] = getGain(entropy, data, column)        %返回值double类型
    [m,n] = size(data);
    feature = data(:, column);
    feature_distinct = unique(feature);
    feature_num = length(feature_distinct);
    feature_proc = cell(feature_num, 2);
    feature_proc(:, 1) = feature_distinct(:, 1);
    feature_proc(:, 2) = num2cell(zeros(feature_num, 1));
    f_entropy = 0;
     for i = 1:feature_num
       feature_row = 0;
       for j = 1:m
           if feature_proc{i, 1} == data{j, column}
               feature_proc{i, 2} = feature_proc{i, 2} + 1;
               feature_row = feature_row + 1;
           end
       end
       feature_data = cell(feature_row,n);
       feature_row = 1;
       for j = 1:m
           if feature_distinct{i, 1} == data{j, column}
               feature_data(feature_row, :) = data(j, :);
               feature_row = feature_row + 1;
           end
       end
       f_entropy = f_entropy + feature_proc{i, 2} / m * getEntropy(feature_data);
    end
    gain = entropy - f_entropy;
end
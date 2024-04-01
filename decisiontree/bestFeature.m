%****************************************
%bestFeature.m
%****************************************
%获取最优划分属性
function [column] = bestFeature(data)       %返回值double类型
    [~,n] = size(data);
    featureSize = n - 1;
    gain_proc = cell(featureSize, 2);
    entropy = getEntropy(data);
    for i = 1:featureSize
        gain_proc{i, 1} = i;
        gain_proc{i, 2} = getGain(entropy, data, i);%ID3
  %      gain_proc{i, 2} = getGainRatio(entropy, data, i);%增益率
    end
    max = gain_proc{1,2};
    max_label = 1;
    for i = 1:featureSize
        if gain_proc{i, 2} > max
            max = gain_proc{i, 2};
            max_label = i;
        end
    end
    column = max_label;
end
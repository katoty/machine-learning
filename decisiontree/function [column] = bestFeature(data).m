function [column] = bestFeature(data)
    [~, n] = size(data);
    featureSize = n - 1;
    gini_proc = zeros(featureSize, 1);
    for i = 1:featureSize
        gini_proc(i) = getGini(data, i);
    end
    [~, column] = max(gini_proc);
end

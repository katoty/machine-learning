%****************************************
%createTree.m
%****************************************
%生成决策树ID3算法
%data：训练集
%feature：属性集
function [node] = createTree(data, feature)
    type = mostType(data);  %cell类型
    [m, n] = size(data);
    %生成节点node
    %value：分类结果，若为null则表示该节点是分支节点
    %name：节点划分属性
    %branch：节点属性值
    %children：子节点
    node = struct('value','null','name','null','branch','null','children',{});
    temp_type = data{1, n};
    temp_b = true;% 表明样本中数据是不是同一类
    
    for i = 1 : m
        if temp_type ~= data{i, n}
            temp_b = false;
        end
    end
    %样本中全为同一分类结果，则node节点为叶节点
    if temp_b == true
        node(1).value = data(1, n); %cell类型
        return;
    end
    %属性集合为空，将结果标记为样本中最多的分类
    if isempty(feature) == 1
        node(1).value = type;  %cell类型
        return;
    end 
    %获取最优划分属性
  %  feature_bestColumn = bestFeature(data); %最优属性列数，double类型
  %  feature_bestColumn = bestFeatureC45(data); %最优属性列数，double类型
  %  feature_bestColumn = bestFeatureCART(data); %最优属性列数，double类型
       feature_bestColumn = bestFeatureCART2(data); %最优属性列数，double类型
%    best_feature = data(:,feature_bestColumn); %最优属性列，cell类型 
%    best_distinct = unique(best_feature); %最优属性取值
    best_distinct = feature(2:end,feature_bestColumn); %最优属性取值
    best_num = length(best_distinct); %最优属性取值个数
    best_proc = cell(best_num, 2);
    best_proc(:, 1) = best_distinct(:, 1);
    best_proc(:, 2) = num2cell(zeros(best_num, 1));
    %循环该属性的每一个值
    for i = 1 : best_num 
        if ~isempty(best_proc{i, 1})
            best_proc{i, 1}
            isempty(best_proc{i,1})
        %为node创建一个bach_node分支，设样本data中该属性值为best_proc(i, 1)的集合为Dv
        bach_node = struct('value', 'null', 'name', 'null', 'branch', 'null', 'children',{});
        Dv_index = 0;
        for j = 1:m
            if data{j, feature_bestColumn} == best_proc{i, 1}
                Dv_index = Dv_index + 1;
            end
        end
        Dv = cell(Dv_index, n);
        Dv_index2 = 1;
        for j = 1:m
            if best_proc{i, 1} == data{j, feature_bestColumn}
                Dv(Dv_index2, :) = data(j, :);
                Dv_index2 = Dv_index2 + 1;
            end
        end
        Dfeature = feature;
        %Dv为空则将结果标记为样本中最多的分类
        if isempty(Dv) == 1
%             bach_node.value = type;
%             bach_node.name = feature(feature_bestColumn);
%             bach_node.branch = best_proc(i, 1);
%             node.children(i) = bach_node;
            bach_node(1).value = type;
            bach_node(1).name = feature(1,feature_bestColumn);
            bach_node(1).branch = best_proc(i, 1);
            node(1).children(i) = bach_node;
            return;
        else
            Dfeature(:,feature_bestColumn) = [];
            Dv(:,feature_bestColumn) = [];
            %递归调用createTree方法
            bach_node = createTree(Dv, Dfeature);
            bach_node(1).branch = best_proc(i, 1);
            %bach_node(1).name = feature(feature_bestColumn);
            bach_node(1).name = feature(1,feature_bestColumn);
            node(1).children(i) = bach_node;
        end
        end
    end
end

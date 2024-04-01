%****************************************
%mostType.m
%****************************************
%����������������
function [type] = mostType(data) %����ֵcell����
    type = cell(1,1);%�������������ǩ
    [m,n] = size(data);
    res = data(:, n);
    res_distinct = unique(res);
    res_num = length(res_distinct);
    res_proc = cell(res_num,2);
    res_proc(:, 1) = res_distinct(:, 1);
    res_proc(:, 2) = num2cell(zeros(res_num,1));
    for i = 1:res_num
        for j = 1:m
            if res_proc{i, 1} == data{j, n}
                res_proc{i, 2} = res_proc{i, 2} + 1;
            end
        end
    end
    if res_num == 1
        type{1,1} = data(:,n);
    else
        if res_proc{1, 2}> res_proc{2, 2}
           type{1,1} = res_proc{1, 1};
       else
        type{1,1} = res_proc{2, 1};
        end
    end
end
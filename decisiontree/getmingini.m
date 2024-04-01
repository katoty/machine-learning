%***************************************
%  给出gini指数最小属性
%****************************************
s = [];
for i = 1:6
    s(i) = getGini(data2,i);
end
[minValue, index] = min(s);
disp(['最小值为: ', num2str(minValue)]);
disp(['最小值序号: ', num2str(index)]);
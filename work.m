Data=xlsread("C:\Users\lty\Downloads\data.xls");
%获取行列数
row=size(Data,1);
column=size(Data,2);
%最后一列加1.5
for i=1:row
    Data(i,8)=Data(i,8)+1.5;
end
tn=3000;

%训练集
train_xdata=Data(1:tn, 1: 7);
train_ydata=Data(1:tn,8);
%给训练集加一列1
for i=1:tn
train_xdata(i,8)=1;
end

%测试集
test_xdata=Data(tn:row ,1 :7);
test_ydata=Data(tn:row , 8);
%给测试集加一列1
for i=1:row-tn+1
test_xdata(i,8)=1;
end

%训练启动
X=train_xdata;
Y=train_ydata;
X_1=test_xdata;
Y_1=test_ydata;
w=inv(X'*X)*X'*Y;
%计算预测值
disp("预测值如下");
predict_y=X_1*w
%计算误差
ebs=Y_1-predict_y;
disp("均方误差如下");
mse=norm(ebs,2)/(row-tn+1)




clear all;
clc;
data_x=rand(10,3);
data_y=[0;0;1;2;0;2;1;2;0;1];
data=[data_x,data_y];
%自助法
[m, n]=size(data);
S=zeros(m,n);
num=randi([1 10],1,10);

for i =1:m
    S(i,:)=data(num(i),:);
end
S
    

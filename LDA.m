function class = LDA(x, xi, xj)
%% LDA分类器
    ui = mean(xi);  % 计算第一个类别的样本均值向量
    uj = mean(xj);  % 计算第二个类别的样本均值向量
    ni = size(xi, 1);  % 第一个类别的样本数
    nj = size(xj, 1);  % 第二个类别的样本数
    sw = (ni-1)*cov(xi) + (nj-1)*cov(xj);  % 类内散度矩阵
    w = sw \ (ui' - uj');  % 计算投影方向
    uyi = w'*ui';
    uyj = w'*uj';
    c = 0.5*(uyi+uyj);
    h = x*w- c;  % 计算投影后的样本值，这里x应为行向量
    if h > 0
        class = true;
    else
        class = false;
    end
end



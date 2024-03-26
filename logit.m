function class = logit(x, xi, yi, xj, yj)
    %% logit分类器
    %% class 为数据类别 x 为待预测数据取值 i 或 j； xi，xj 为第 i，j 类数据 yi，yj 为对应的类别。
    X = [xi; xj];
    [m, ~] = size(yi);
    [n, ~] = size(yj);

    %% 令第 i 类为 +，第 j 类为 -
    yi = ones(m, 1);
    yj = zeros(n, 1);
    Y = [yi; yj];

    [m, n] = size(X);
    X_enlarge = [X ones(m, 1)];
    
    %% BELTA1 作为 (w,b) 初值 ,alpha 学习率, internum 迭代次数
    belta1 = rand(n + 1, 1);
    alpha = 0.001;
    internum = 100;
    for i = 1:internum
        Z = X_enlarge * belta1;
        P = 1 ./ (1 + exp(-Z));
        E = P - Y;
        belta2 = belta1 - alpha * X_enlarge' * E;
        belta1 = belta2;
    end
   

    %% 计算 x 预测值当 p > 0.5 认为它是 +，p < 0.5 认为它是 -。
    x_enlarge = [x 1];
    z = x_enlarge * belta1;
    p = 1 / (1 + exp(-z));
    class = p>=0.5;
end







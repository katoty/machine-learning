%****************************************
%drawTree.m
%****************************************
% 画出决策树
function [] = drawTree(node)
    % 遍历树
    nodeVec = [];
    nodeSpec = {};
    edgeSpec = [];
    [nodeVec,nodeSpec,edgeSpec,~] = travesing(node,0,0,nodeVec,nodeSpec,edgeSpec);
    treeplot(nodeVec);
    [x,y] = treelayout(nodeVec);
    [~,n] = size(nodeVec);
    x = x';
    y = y';
    text(x(:,1),y(:,1),nodeSpec,'FontSize',15,'FontWeight','bold','VerticalAlignment','bottom','HorizontalAlignment','center','FontName','Microsoft YaHei UI');
    x_branch = [];
    y_branch = [];
    for i = 2:n
        x_branch = [x_branch; (x(i,1)+x(nodeVec(i),1))/2];
        y_branch = [y_branch; (y(i,1)+y(nodeVec(i),1))/2];
    end
    text(x_branch(:,1),y_branch(:,1),edgeSpec(1,:),'FontSize',12,'Color','blue','FontWeight','bold','VerticalAlignment','bottom','HorizontalAlignment','center','FontName','Microsoft YaHei UI');
end

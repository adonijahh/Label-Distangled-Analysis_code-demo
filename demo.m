%% LDA复现：分类器：KNN
% N. Xiao, L. Zhang, X. Xu, T. Guo, and H. Ma, “Label Disentangled Analysis for unsupervised visual domain adaptation,” 
% Knowl.-Based Syst., vol. 229, p. 107309, Oct. 2021, doi: 10.1016/j.knosys.2021.107309.
% 复现：Y. Tian, University of Electronic Science and Technology of China
clear;
clc;
close all;
%% 数据准备
load Gas_data
load A
data1 = A;
%% 输入
Xs = mapminmax(data1{1}',0,1);              % 源域数据集 d*ns
Xt = mapminmax(data1{2}',0,1);              % 目标域数据集 d*nt
X = [Xs Xt];                                % 拼接出的数据矩阵 d*ns+nt
Ys = labels1{1};                            % 标准源域标签
actual_Yt = labels1{2};                     % 实际目标域标签
Ys_onehot = onehot(Ys)';                    % 源域标签 c*n
ns = size(Xs,2);                            % 源域样本数量
nt = size(Xt,2);                            % 目标域样本数量

max_epoch = 1000;                             % 最大迭代次数
goal = 0.5;                               % 收敛目标

dim = 5;                                    % 降维维度数
knn_num = 2;                                % knn样本个数
mu = 2;                                     % 平衡参数
lambda = 0.01;                              % 正则化系数

P = rand(size(Ys_onehot,1)+size(X,1),dim);  % 投影方向P (c+m)*dim

%% Initialize target label-score matrix 预训练分类器 这里采用KNN
mdl = fitcknn((P'*[Xs;Ys_onehot])',Ys ,'NumNeighbors',knn_num);
pseudo_xt = predict(mdl,(P'*[Xt;zeros(size(Ys_onehot,1),size(Xt,2))])');
Yt_hat = onehot(pseudo_xt)';                % 伪标签
%% Initialize label matrix 初始化标签矩阵
Y = [Ys_onehot Yt_hat];
%% Construct adaptation matrix 构建自适应矩阵A
% 构建A0
for i = 1:1:ns+nt
    for j = 1:1:ns+nt
        if i <= ns && j<= ns
            A0(i,j) = 1/(ns*ns);
        elseif i >= ns+1 && j>= ns+1
            A0(i,j) = 1/(nt*nt);
        else
            A0(i,j) = -1/(ns*nt);
        end
    end
end
% 构建Ac
n_s_c = tabulate(Ys);                       % 源域各类标签分类数量
n_s_c = n_s_c(:,2)';
if size(n_s_c,2)<6                          % 补0，以保证六类
    n_s_c = [n_s_c,0];
end
n_t_c = tabulate(pseudo_xt);                % 目标域各类伪标签数量
n_t_c = n_t_c(:,2)';
if size(n_t_c,2)<6                          % 补0，以保证六类
    n_t_c = [n_t_c,0];
end
% 计算边缘自适应矩阵A0
for i = 1:1:ns+nt
    for j = 1:1:ns+nt
        if i <= ns && j<= ns
            if Ys(i)==Ys(j)
                Ac(i,j) = 1/(n_s_c(Ys(i))*n_s_c(Ys(i)));
            end
        elseif i >= ns+1 && j>= ns+1
            if pseudo_xt(i-ns)==pseudo_xt(j-ns)
                Ac(i,j) = 1/(n_s_c(pseudo_xt(i-ns))*n_s_c(pseudo_xt(j-ns)));
            end
        elseif i <= ns && j>= ns+1
            if Ys(i)==pseudo_xt(j-ns)
                Ac(i,j) = -1/(n_s_c(Ys(i))*n_t_c(pseudo_xt(j-ns)));
            end
        elseif i >= ns+1 && j<= ns
            if pseudo_xt(i-ns)==Ys(j)
                Ac(i,j) = -1/(n_s_c(Ys(j))*n_t_c(pseudo_xt(i-ns)));
            end
        end
    end
end
A = A0+Ac;
% Construct centering matrix 计算定心矩阵
n = ns+nt;
H = eye(n)-1/n*ones(n,n);
epoch = 0;
error = Inf;
%% Solve the Eq. (14) to update P 训练计算投影方向P
while epoch <= max_epoch && error > goal
    epoch = epoch + 1;
    disp(['Now Epoch:',num2str(epoch)]);
    term = [X;mu*Y];
    loss = pinv((term*H*term')')*(term*A*term'+lambda*eye(size(Xs,1)+6));               % 计算优化目标
    [eig_vectors,eig_value] = eig(loss);                                                % 特征值分解
    [D_sort,eig_index] = sort(diag(eig_value),'descend');
    sort_eig_vector = eig_vectors(:,eig_index');
    P_star = sort_eig_vector(:,end-dim+1:end);                                          % 目标投影方向
    % Update transformed data matrix 更新投影数据矩阵
    X_star = P_star'*[X;Y];
    % Train a existing classifier 重新训练分类器以得到数据伪标签
    mdl = fitcknn((P_star'*[Xs;Ys_onehot])',Ys ,'NumNeighbors',knn_num);
    pseudo_xt = predict(mdl,(P_star'*[Xt;zeros(size(Ys_onehot,1),size(Xt,2))])');
    Yt_hat = onehot(pseudo_xt)';        % 伪标签
    Y = [Ys_onehot Yt_hat];
    % Update A by Eq. (9) and Eq. (11); 重新计算Ac，以重新计算A
    n_t_c = tabulate(pseudo_xt);        % 重新计算目标域各类伪标签数量
    n_t_c = n_t_c(:,2)';
    if size(n_t_c,2)<6
        n_t_c = [n_t_c,0];
    end
    for i = 1:1:ns+nt
        for j = 1:1:ns+nt
            if i <= ns && j<= ns
                if Ys(i)==Ys(j)
                    Ac(i,j) = 1/(n_s_c(Ys(i))*n_s_c(Ys(i)));
                end
            elseif i >= ns+1 && j>= ns+1
                if pseudo_xt(i-ns)==pseudo_xt(j-ns)
                    Ac(i,j) = 1/(n_s_c(pseudo_xt(i-ns))*n_s_c(pseudo_xt(j-ns)));
                end
            elseif i <= ns && j>= ns+1
                if Ys(i)==pseudo_xt(j-ns)
                    Ac(i,j) = -1/(n_s_c(Ys(i))*n_t_c(pseudo_xt(j-ns)));
                end
            elseif i >= ns+1 && j<= ns
                if pseudo_xt(i-ns)==Ys(j)
                    Ac(i,j) = -1/(n_s_c(Ys(j))*n_t_c(pseudo_xt(i-ns)));
                end
            end
        end
    end
    A = A0+Ac;
    
    % 计算误差 误差函数：mse
    pseudo_xs = predict(mdl,(P_star'*[Xs;zeros(size(Ys_onehot,1),size(Xs,2))])');
    dif = Ys - pseudo_xs;
    error = mse(dif);
    rep_error = ['Epoch: ',num2str(epoch),'      Error:',num2str(error)];
    disp(rep_error);
end
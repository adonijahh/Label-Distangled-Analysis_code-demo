%% LDA���֣���������KNN
% N. Xiao, L. Zhang, X. Xu, T. Guo, and H. Ma, ��Label Disentangled Analysis for unsupervised visual domain adaptation,�� 
% Knowl.-Based Syst., vol. 229, p. 107309, Oct. 2021, doi: 10.1016/j.knosys.2021.107309.
% ���֣�Y. Tian, University of Electronic Science and Technology of China
clear;
clc;
close all;
%% ����׼��
load Gas_data
load A
data1 = A;
%% ����
Xs = mapminmax(data1{1}',0,1);              % Դ�����ݼ� d*ns
Xt = mapminmax(data1{2}',0,1);              % Ŀ�������ݼ� d*nt
X = [Xs Xt];                                % ƴ�ӳ������ݾ��� d*ns+nt
Ys = labels1{1};                            % ��׼Դ���ǩ
actual_Yt = labels1{2};                     % ʵ��Ŀ�����ǩ
Ys_onehot = onehot(Ys)';                    % Դ���ǩ c*n
ns = size(Xs,2);                            % Դ����������
nt = size(Xt,2);                            % Ŀ������������

max_epoch = 1000;                             % ����������
goal = 0.5;                               % ����Ŀ��

dim = 5;                                    % ��άά����
knn_num = 2;                                % knn��������
mu = 2;                                     % ƽ�����
lambda = 0.01;                              % ����ϵ��

P = rand(size(Ys_onehot,1)+size(X,1),dim);  % ͶӰ����P (c+m)*dim

%% Initialize target label-score matrix Ԥѵ�������� �������KNN
mdl = fitcknn((P'*[Xs;Ys_onehot])',Ys ,'NumNeighbors',knn_num);
pseudo_xt = predict(mdl,(P'*[Xt;zeros(size(Ys_onehot,1),size(Xt,2))])');
Yt_hat = onehot(pseudo_xt)';                % α��ǩ
%% Initialize label matrix ��ʼ����ǩ����
Y = [Ys_onehot Yt_hat];
%% Construct adaptation matrix ��������Ӧ����A
% ����A0
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
% ����Ac
n_s_c = tabulate(Ys);                       % Դ������ǩ��������
n_s_c = n_s_c(:,2)';
if size(n_s_c,2)<6                          % ��0���Ա�֤����
    n_s_c = [n_s_c,0];
end
n_t_c = tabulate(pseudo_xt);                % Ŀ�������α��ǩ����
n_t_c = n_t_c(:,2)';
if size(n_t_c,2)<6                          % ��0���Ա�֤����
    n_t_c = [n_t_c,0];
end
% �����Ե����Ӧ����A0
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
% Construct centering matrix ���㶨�ľ���
n = ns+nt;
H = eye(n)-1/n*ones(n,n);
epoch = 0;
error = Inf;
%% Solve the Eq. (14) to update P ѵ������ͶӰ����P
while epoch <= max_epoch && error > goal
    epoch = epoch + 1;
    disp(['Now Epoch:',num2str(epoch)]);
    term = [X;mu*Y];
    loss = pinv((term*H*term')')*(term*A*term'+lambda*eye(size(Xs,1)+6));               % �����Ż�Ŀ��
    [eig_vectors,eig_value] = eig(loss);                                                % ����ֵ�ֽ�
    [D_sort,eig_index] = sort(diag(eig_value),'descend');
    sort_eig_vector = eig_vectors(:,eig_index');
    P_star = sort_eig_vector(:,end-dim+1:end);                                          % Ŀ��ͶӰ����
    % Update transformed data matrix ����ͶӰ���ݾ���
    X_star = P_star'*[X;Y];
    % Train a existing classifier ����ѵ���������Եõ�����α��ǩ
    mdl = fitcknn((P_star'*[Xs;Ys_onehot])',Ys ,'NumNeighbors',knn_num);
    pseudo_xt = predict(mdl,(P_star'*[Xt;zeros(size(Ys_onehot,1),size(Xt,2))])');
    Yt_hat = onehot(pseudo_xt)';        % α��ǩ
    Y = [Ys_onehot Yt_hat];
    % Update A by Eq. (9) and Eq. (11); ���¼���Ac�������¼���A
    n_t_c = tabulate(pseudo_xt);        % ���¼���Ŀ�������α��ǩ����
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
    
    % ������� ������mse
    pseudo_xs = predict(mdl,(P_star'*[Xs;zeros(size(Ys_onehot,1),size(Xs,2))])');
    dif = Ys - pseudo_xs;
    error = mse(dif);
    rep_error = ['Epoch: ',num2str(epoch),'      Error:',num2str(error)];
    disp(rep_error);
end
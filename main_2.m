clear
m = 128;
n = 256;
Success_count_OMP = zeros(63,1);
Success_count_SP = zeros(63,1);
Success_count_IHT = zeros(63,1);
for S = 3:3:63
    for iter = 1:500
        A = randn(m, n);
        A = normc(A);
        support_indices = randsample(n, S);
        x = zeros(n, 1);
        x(support_indices) = randn(S, 1);  % N(0,1) 分布的非零元素
        y = A * x;
        xOMP = orthogonal_matching_pursuit(A, y, S);
        xSP = subspace_pursuit(A, y, S);
        xIHT = iterative_hard_thresholding(A, y, S);
        errorOMP = norm(x - xOMP)/norm(x);
        errorSP = norm(x - xSP)/norm(x);
        errorIHT = norm(x - xIHT)/norm(x);

        if errorOMP < 1e-6
            Success_count_OMP(S) = Success_count_OMP(S)+1;
        end

        if errorSP < 1e-6
            Success_count_SP(S) = Success_count_SP(S)+1;
        end

        if errorIHT < 1e-6
            Success_count_IHT(S) = Success_count_IHT(S)+1;
        end
        clear A x errorOMP errorSP errorIHT xOMP xSP xIHT
    end
    fprintf('Finish a S : %d\n',S)
end
Success_rate_OMP = Success_count_OMP/500;
Success_rate_SP = Success_count_SP/500;
Success_rate_IHT = Success_count_IHT/500;
indices = 3:3:63;

figure;
plot(indices, Success_rate_OMP(indices), '-o', 'DisplayName', 'OMP');
hold on;
plot(indices, Success_rate_SP(indices), '-s', 'DisplayName', 'SP');
plot(indices, Success_rate_IHT(indices), '-d', 'DisplayName', 'IHT');
title('Comparision Between Three Algorithms');
xlabel('S');
ylabel('Success  rate');
legend('show');
grid on;




%OMP
function x_OMP = orthogonal_matching_pursuit(A, y, S)
    n = 256;
    x = zeros(n, 1);%初始化x
    y_r = y; %初始化y的残差
    A_T = A';%A的转置
    s = [];
    for iter_1 = 1:S
        x_input_r = A_T*y_r;
        s = union(s, find(Hard_Thresholding_Function(x_input_r,1,n)));%S = S与支持集的并集
        A_s = A(:,s);
        x_s= pinv(A_s)*y;
        expand_x = zeros(n,1);
        expand_x(s) = x_s;
        y_r = y-A*expand_x;
    end
    x_OMP = expand_x;
end
%Hs()
function x_Hard_Thresholding = Hard_Thresholding_Function(x,S,n)
    x_pre = x; %x的初始值
    x_abs = abs(x); %x的绝对值
    [x_abs,i] = sort(x_abs,'descend');
    threshold = x_abs(S);
    for iter = 1:n
        if abs(x_pre(iter)) < threshold
            x_pre(iter) = 0;
        end
    end
    x_Hard_Thresholding = x_pre;
end



%SP
function x_SP = subspace_pursuit(A, y, S)
    n = 256;
    A_T = A'; %A的转置
    s = find(Hard_Thresholding_Function(A_T*y, S, n));  % Initialization
    A_s = A(:,s);
    y_r = resid(y,A_s);
    min = 1e-6;
    epoch = 0;
    while 1
        s_1 = union(s,find(Hard_Thresholding_Function(A_T*y_r, S, n)));%(Expand support)
        expand_b = zeros(n,1);  %init b
        A_s_1 = A(:,s_1);     
        b_s_1= pinv(A_s_1)*y;   
        expand_b(s_1) = b_s_1;  %Estimate 2S-sparse signal

        s = find(Hard_Thresholding_Function(expand_b, S, n));  %Shrink support  
        expand_x = zeros(n,1);
        A_s = A(:,s);
        x_s= pinv(A_s)*y;   
        expand_x(s) = x_s;  %(Estimate S-sparse signal

        y_r = y-A*expand_x; %(Compute estimation error
        epoch = epoch+1;

        if norm(y_r) < min %循环退出条件
            break
        end

        if epoch > 20000
            break
        end
    end
    x_SP = expand_x;
end

%resid
function y_resid =  resid(y,A)
    y_p = A*pinv(A)*y; %计算投影
    y_r= y-y_p;        %计算残差
    y_resid = y_r;  
end
%IHT
function x_IHT = iterative_hard_thresholding(A, y, S)
    n = 256;
    min = 1e-16;%迭代中预测的x与前一轮的x的范数相差min时退出循环
    l_0 = 0.1; %初始学习率,若不添加，会导致梯度过大而无法收敛
    expand_x = zeros(n,1);
    epoch = 0;
    while 1
        x_pre = expand_x;
        x_input = expand_x + l_0*(A')*(y-A*expand_x);  %x+AT(y Ax)
        expand_x = Hard_Thresholding_Function(x_input, S, n);
        epoch = epoch+1;
        if norm(expand_x-x_pre) < min  %当预测的x与前一轮的x的范数相差为min退出
            break
        end
        if epoch > 20000
            break
        end
    end
    x_IHT = expand_x;
end



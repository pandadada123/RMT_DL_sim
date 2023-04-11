% Multi-layer network, linear/nonlinear, with/without batch normalization

% Ref:
% [1] H. Daneshmand, J. Kohler, F. Bach, T. Hofmann, and A. Lucchi, "Batch Normalization Provably Avoids
% Rank Collapse for Randomly Initialised Deep Networks," June 2020.
% [2] J. Bjorck, C. Gomes, B. Selman, and K. Q. Weinberger, "Understanding Batch Normalization," Nov. 2018.
% [3] H. Daneshmand, A. Joudaki, and F. Bach, "Batch Normalization Orthogonalizes Representations in Deep
% Random Networks," June 2021.
% [4] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal
% Covariate Shift," Mar. 2015.


close all
clear all

M = 100:200:1900; % different width of matrices
% M = 1000;

%% Rank versus the number of hidden layers, with/without batch normalization 
Nlayer=50;  % number hidden layers
Rank_last=[];

%% Rank versus width of linear net, with/without batch normalization 
for m=M
%% Dimensions of matrices
c0=2; % layer0, c0=m/n0   
% c0=1; % square case
n0=m/c0; % layer0

c=ones(1,Nlayer);  % ratios of hidden layers

n=zeros(1,Nlayer);
n(1)=n0/c(1); %layer1
for tt=2:Nlayer
    n(tt)=n(tt-1)/c(tt);
end

%% All eig with 100 rep
L0=[];  % eig of the data layer
L=cell(1,Nlayer);  % to save all the eigenvalues in ten repitions for each layer.

N_rep=100;  
for rep=1:N_rep

    X0=randn(n0,m); % input layer
    % X0=BN(X0);    % should be no BN if compared with nonlinear function
    Y0=X0*X0.'/n0;  
    L0=[L0, eig(Y0)]; % eig of the input layer

    W=cell(1,Nlayer);
    X=cell(1,Nlayer);
    Y=cell(1,Nlayer);

    W0=randn(n(1),n0); % for the first hidden layer
    X{1}=act (BN( 1/sqrt(n0)  * W0*X0 ));  
    Y{1}=X{1}*X{1}.'/n(1);
    L{1}=[L{1}, eig(Y{1})];  % eig of the first hidden layer

    for k=2:Nlayer
        W{k-1}=randn(n(k),n(k-1));
        X{k}=act (BN( 1/sqrt(n(k-1)) * W{k-1}*X{k-1}));
        Y{k}=X{k}*X{k}.'/n(k);
        L{k}=[L{k}, eig(Y{k})];   % eig of the k-th hidden layer
    end
end

%% Settings of histogram
Threshold=0.1;  % threshhold is always 0.1
binwidth=0.1; 
Max = 5;
% edges = Threshold-binwidth : binwidth : Max;
edges = 0 : binwidth : Max;

%% Rank/histogram/probability of the input (data) layer
figure
H0=histogram(L0,edges);
counts0=H0.Values/n0;
counts0=counts0/binwidth;

counts0_avg=counts0/N_rep; % divided by 10 rep

Pr_zero_0 = counts0_avg(1)*binwidth
Pr_nonzero_0 = sum(counts0_avg(2:end)*binwidth)
Rank0=Pr_nonzero_0*n0;
close

figure  % histogram of the input layer
histogram('BinEdges',edges,'BinCounts',counts0_avg)
title('The input layer')
close 

Pr_nonzero0_theo=curve(0,Threshold);
Rank0_theo=Pr_nonzero0_theo*n0;

% directly from the matrix
Rank0_3=numel(find(L0>=Threshold))/N_rep;

%% Rank/histogram/probability of each hidden layer
Pr_nonzero_all=[];
Rank=[];
Rank=Rank0;
Rank_theo=[];
Rank_theo=Rank0_theo;
Rank_3=[];
Rank_3=Rank0_3;  % initialization by the input layer
for kk=1:Nlayer
    figure
    H=histogram(L{kk},edges);
    counts=H.Values/n(kk); 
    counts=counts/binwidth;   

    counts_avg=counts/N_rep;  % divided by 10 rep

    Pr_zero = counts_avg(1)*binwidth
    Pr_nonzero = sum(counts_avg(2:end)*binwidth)
    close

    figure    % histogram for the kk-th hidden layer
    histogram('BinEdges',edges,'BinCounts',counts_avg)
    title(sprintf('The %dth hidden layer',kk)) 
    hold on

    Rank=[Rank Pr_nonzero*n(kk)];  % save the rank for each layer
    close

    Pr_nonzero_theo=curve(kk,Threshold);
    Rank_theo=[Rank_theo Pr_nonzero_theo*n(kk)]; % theoretical rank from theoretical curve
    
    % directly from the matrix
    Rank_3=[Rank_3 numel(find(L{kk}>=Threshold))/N_rep];
end
%% Rank versus width of linear net, with/without batch normalization 
Rank_last=[Rank_last Rank_3(end)];  % Rank of the last hidden layer (for different width)

end

%% Rank/Probability theoretical curves for square linear case
function Pr_nonzero_theo=curve(Nlayer,Threshold)
M=Nlayer+1;
phi=0.01:1e-5:pi/(M+1)-0.01;  
% fplot(@(x) (sin(x))^2*(sin(5*x))^4/((sin(6*x))^5)/pi)
x=sin((M+1)*phi).^(M+1)./(sin(phi).*(sin(M*phi).^M));
f=1./(pi*x).*sin((M+1)*phi)./sin(M*phi).*sin(phi);
% plot(x,f,'r','LineWidth',2)
xx=fliplr(x);
ff=fliplr(f);
index=find(xx>=Threshold);
xxx=xx(index);
fff=ff(index);
Pr_nonzero_theo = simps(xxx,fff);
end

%% Nonlinear activation function (with/without)
function Y2=act(Y)
syms x

fun = @(x) x; % Lineat net
c1=1;

% fun = @(x) max(0,x); % ReLU net
% c1=sqrt(2);

Y2=c1*fun(Y); % substitute in point-wise data, and normalization of f, return Y2
end

%% Batch normalization (with/without)
function Y=BN(M)  
%% with BN
    d=size(M,1);  % get the first dim of M

    M2=M*M.';   % diagonal elements are approx n, off-diagonal less than n
    
    v = zeros(1, length(M2));
    for k = 1 : length(M2)
        v(k) = M2(k,k);
    end
    Y1=diag(v);
    t =Y1^(-1/2);
    Y=t*M;  % 对角矩阵的每一项都是标准差

% % strict BN
%     Y=(M*M.')^(-1/2)*M;

%% rescale Y to make it comparable to the linear case
    Y=sqrt(d)*Y;
%% without BN 
% Y=M; 
end

% Nonlinear activation functions 

% Ref:
% [1] J. Pennington and P. Worah, "Nonlinear random matrix theory for deep learning," in Advances in Neural
% Information Processing Systems, 2017.

close all
clear all

Nlayer=1; % number of hidden layers

%% Dimensions of matrices
c0=1; % layer0
c=ones(1,Nlayer); % square matrices

m=1000;
n=zeros(1,Nlayer);
n0=m/c0; % layer0
n(1)=n0/c(1); % layer1

%% All eig with 10 rep
N_rep=10;
Threshold=0.1;
binwidth=0.1;
Max = 5;
edges = Threshold-binwidth : binwidth : Max;
counts_all=0;
counts0_all=0;

L_all=[];
for rep=1:N_rep

    X0=randn(n0,m); % input layer
    Y0=X0*X0.'/n0;

    W=cell(1,Nlayer);
    X=cell(1,Nlayer);
    Y=cell(1,Nlayer);

    W{1}=randn(n(1),n0);
    X{1}=act(W{1}*X0/sqrt(n0)); % apply activation function to the first hidden layer
    Y{1}=X{1}*X{1}.'/n(1);

    %% eig of the input layer
    L0=eig(Y0); % eig of the data layer
    H0=histogram(L0,edges);
    counts0=H0.Values/n0;
    counts0=counts0/binwidth;
    counts0_all=counts0_all+counts0;
    close
    
    %% eig of the first hiden layer
    L=eig(Y{Nlayer}); 
    L=real(L); % not really need this
    % L_all=[L_all L];
    
    H=histogram(L,edges);
    counts=H.Values/n(Nlayer);
    counts=counts/binwidth;
    
    counts_all=counts_all+counts;  
    
end
counts0_avg=counts0_all/N_rep; % input layer
counts_avg=counts_all/N_rep; % first hidden layer

Pr_zero_0 = counts0_avg(1)*binwidth
Pr_nonzero_0 = sum(counts0_avg(2:end)*binwidth)
Rank0=Pr_nonzero_0*n0;


%% histogram of the input layer
figure  % eig for data layer
Pr_zero = counts0_avg(1)*binwidth
Pr_nonzero = sum(counts0_avg(2:end)*binwidth)

histogram('BinEdges',edges,'BinCounts',counts0_avg)
title('The input layer')
hold on

%% histogram of the first hidden layer
figure
Pr_zero = counts_avg(1)*binwidth
Pr_nonzero = sum(counts_avg(2:end)*binwidth)
Rank = Pr_nonzero*1000;

histogram('BinEdges',edges,'BinCounts',counts_avg)
title('The first hidden layer')
hold on

%% nonlinear activation function
function Y2=act(Y)
% Y2=Y;
% Y2=abs(Y);
% Y2=abs(Y)-3*sqrt(2/pi);
% Y2=extractdata(relu(dlarray(Y)));
% Y2=extractdata(leakyrelu(dlarray(Y),0.05));
% Y2=extractdata(sigmoid(dlarray(Y)));
% Y2=tanh(Y);
% Y2=abs(Y)-sqrt(2/pi);

syms x
% fun = @(x) x; % to check this method is correct
% fun = @(x) abs(x);
% fun = @(x) abs(x)-sqrt(2/pi);
% fun = @(x) abs(x-1)-sqrt(2/pi); % mean not equal to 0

% fun = @(x) -1+sqrt(5)*exp(-2*x.^2); 
% fun = @(x) sin(2*x)+cos(3*x/2)-2*exp(-2)*x-exp(-9/8);

fun = @(x) (1-4/sqrt(3).*exp(-x.^2/2)).*erf(x);
% fun = @(x) (1-4/sqrt(3).*exp(-x.^2/2)).*(erf(x)-x);  
% fun = @(x) (1-4/sqrt(3).*exp(-x.^2/2)).*(erf(x)-10000000*x);

% fun = @(x) (1-4/sqrt(3).*exp(-x.^2/2)).*(erf(x))+0.3*x;  % much more
% linear, not preserving the spcetrum

eta = gaussian_eta(fun)
c1 = 1/sqrt(eta)  % normalization factor

syms x
f_norm=eval(['@(x)' char((c1*fun(x)))]);
eta_check = gaussian_eta (  f_norm  )    % check this normalization, should be one.

Y2=c1*fun(Y); % substitute in point-wise data, and normalization of f, return Y2
   
f_mean=gaussian_mean(f_norm) % should be zero
zeta=gaussian_zeta(f_norm) % approximate to zero if spectrum is preserved 
end

%% calculate eta (var of f)
function eta=gaussian_eta(fun) % calculate the Gaussian integral and then normalize it
%     fun = @(x) exp(-x.^2).*log(x).^2;
%     int = integral(fun,-Inf,Inf)
    min = -1e4;
    max = 1e4;
    xx=min:0.05:max;
    syms x
    f00 = eval(['@(x)' char(fun(x).^2)]);
    
    f0=zeros(1,length(xx));
    for xxx=1:length(xx)
%         f0(xxx)=subs(sym(f00),x,xx(xxx)) % symbolic is slow.
        f0(xxx)=f00(xx(xxx));   % Not suitable for caculating ReLU
    end
    
    gaussian=exp(-xx.^2/2)/sqrt(2*pi);
    f=f0.*gaussian;
    eta = simps(xx,f);  % Simpson's numerical integration.
end

%% check the mean of f
function f_mean=gaussian_mean(fun)
    min = -1e4;
    max = 1e4;
    xx=min:0.05:max;
    syms x
    f00 = eval(['@(x)' char(fun(x))]);
    
    f0=zeros(1,length(xx));
    for xxx=1:length(xx)
        f0(xxx)=f00(xx(xxx));
    end
    
    gaussian=exp(-xx.^2/2)/sqrt(2*pi);
    f=f0.*gaussian;
    f_mean = simps(xx,f);
end

%% calculate zeta
function zeta=gaussian_zeta(fun)
    min = -1e4;
    max = 1e4;
    xx=min:0.05:max;
    syms x
    f00 = eval(['@(x)' char(diff(fun(x)))]);
    
    f0=zeros(1,length(xx));
    for xxx=1:length(xx)
        f0(xxx)=f00(xx(xxx));
    end
    
    gaussian=exp(-xx.^2/2)/sqrt(2*pi);
    f=f0.*gaussian;
    zeta = (simps(xx,f))^2;
end

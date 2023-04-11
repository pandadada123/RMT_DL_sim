% Marchěnko-Pastur Law

% Ref: 
% [1] J. Ge, Y.-C. Liang, Z. Bai, and G. Pan, "Large-dimensional random matrix theory and its applications in
% deep learning and wireless communications," Random Matrices: Theory and Applications, vol. 10, p. 2230001,
% Oct. 2021.
% [2] O. Lévêque, "Week 12: Marchenko-pastur's theorem: Stieltjes transform method," 2011.
% [3] M. Potters and J.-P. Bouchaud, A First Course in Random Matrix Theory: for Physicists, Engineers and
% Data Scientists. Cambridge University Press, 1 ed., Nov. 2020.

close all
clear all

%% Size of the matrix 
n=1000;
c=4;% ratio 
m=n*c;

%% Get repetitive eigenvalues 
N_rep=100;
counts_all=0;
L_all=[];

for rep=1:N_rep  

    h=randn(n,m);
    x=h*h.'; 
    x=x/n; 

    L=eig(x);
    L_all=[L_all L];
    
end

%% Histogram
Max = 10;
binwidth = 0.1;
edges = 0 : binwidth : Max;

figure
H0=histogram(L_all,edges);
counts0=H0.Values/n;
counts0=counts0/binwidth;

counts0_avg=counts0/N_rep; % average numbers of histograms 
figure
histogram('BinEdges',edges,'BinCounts',counts0_avg)
% title('MP law')
hold on

%% Theoretical curve of MP-Law (without Dirac function)
a=(1-sqrt(c))^2;  
b=(1+sqrt(c))^2; 
x_axe=a:0.05:b;
f=sqrt((b-x_axe).*(x_axe-a))./(2*pi*x_axe); 
plot(x_axe,f,'r','LineWidth',1.5);
hold on

check_sum=sum((f(2:end)+f(1:end-1))/2*(x_axe(2)-x_axe(1))) % should be 1 if c>1; c if c<1

%% Numerical calculation of theoretical curve
Lam=linspace(a,b,100);
ep=1e-9;
f=zeros(1,length(Lam));
for k=1:length(Lam)
    lam=Lam(k);
    z=lam+1i*ep;
    syms g 
    eqn = z*g^2+(z+1-c)*g+1 == 0;
    assume(imag(g) >= 0); % Add restriction
    S = solve(eqn,g);    
%     f(k)=(1/pi)*imag(S(2));
    f(k)=(1/pi)*imag(S(1));
end

plot(Lam,f,'y--','LineWidth',1);
hold on

check_sum=sum((f(2:end)+f(1:end-1))/2*(Lam(2)-Lam(1))) % should be 1 if c>1; c if c<1

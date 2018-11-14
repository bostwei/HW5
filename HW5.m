%*************************************************************************
% Econ 899 HW5 
% Shiyan Wei
% 11/06/2018
% ************************************************************************

% ************************************************************************
% This script is used to solving the homework 5 for Econ 899
% ***********************************************************************
clear
clc
close all

%% Data Initiated 
N = 66;
n = 0.011;
% Grid Number
Na = 200;

% Initial asset holding
a1 = 0;

% worker retire
JR = 46;

% Benefits

% labor income tax
ttheta  = 0.11;

ggama = 0.42;
ssigma = 2;

% idiosyncratic productivity
z = [3;0.5];
% The probability of productivity
Pz = [0.2037;0.7963];

% Trhansition probability
Pi = [0.9261 1-0.9261;
     1-0.9811 0.9811];

% capital share
aalpha = 0.36;
% depreciation rate
delta = 0.06;

% Asset Space
alb = 0;
aub = 5;



A = linspace(alb, aub, Na)';

a = A* ones(1,Na);
aa = A* ones(1,Na); 

% Import labor effciency
ef = importdata('ef.txt');

%% Dynamic Programming Problem
w = 1.05;
r = 0.05;
b = 0.2;

bbeta = 0.97;

%------------- The utility for the retireed agent ----------------
% consumption of returement
 
c_r = (1+r)*a + b - aa';
c_r(find(c_r <=0)) = NaN;

% utility of the retirement
u_r = (c_r .^ ((1-ssigma) .* ggama))./(1-ssigma);
u_r(find(isnan(u_r))) = -inf;

%----------- The value function for retired agent ----------------

v0_r = zeros(Na,N-JR);
v1_r = zeros(Na,N-JR);
dec_r = zeros(Na,N-JR);

% find the value function for the T age agent
v1_r(:,N-JR)= u_r(:,1); % The coloumn 1 means the agent choose not to save in this period.
v0_r = v1_r;

% find the value function for the j<T age agent
for j = N-JR-1:-1:1
    w_r = u_r + bbeta .*v0_r(:,j+1)';
    [v1_r(:,j), dec_r(:,j)] = max(w_r,[],2);
    v0_r = v1_r;
end

plot(A,v1_r(:,4))

%----------- The utility function for worked agent ----------------
% labor efficiency 
% - first row is ef if it get z = 3
% - second row is the ef if it get z = 0.5
e = z * ef';

v0_w_zh = zeros(Na,JR);
v0_w_zl = zeros(Na,JR);
v1_w_zh = zeros(Na,JR);
v1_w_zl = zeros(Na,JR);
% the last period of value function of working agent is the same as the
% retire people

v0_w_zh(:,JR) = v0_r(:,1);
v0_w_zl(:,JR) = v0_r(:,1);
v1_w_zh = v0_w_zh;
v1_w_zl = v0_w_zl;

% searh for the best l an aa' combination for each a
for ai = 1:Na
    
end


dec_w_zh = zeros(Na,JR-1);
dec_w_zl = zeros(Na,JR-1);

for j = JR-1:-1:1
    
% ------------ the labor choice of worker --------------------




l_zh = (ggama*(1-ttheta) * e(1,j) * w - (1-ggama)*((1+r)*a - aa'))/((1-ttheta)* w *e(1,j)); 
l_zl = (ggama*(1-ttheta) * e(2,j) * w - (1-ggama)*((1+r)*a - aa'))/((1-ttheta)* w *e(2,j));
l_zh(find(l_zh <0)) = NaN;
l_zl(find(l_zl <0)) = NaN;

% consumption of worker
    
  c_w_zh = w * (1-ttheta)* e(1,j)* l_zh + (1+r) * a - aa';
  c_w_zl = w * (1-ttheta)* e(2,j)* l_zl + (1+r) * a - aa';
  
% utility of the worker is 
 u_w_zh = (c_w_zh.^ggama.*(1-l_zh).^(1-ggama)).^(1-ssigma)/(1-ssigma);
 u_w_zl = (c_w_zl.^ggama.*(1-l_zl).^(1-ggama)).^(1-ssigma)/(1-ssigma);
 u_w_zh(find(isnan(u_w_zh))) = -inf;
 u_w_zl(find(isnan(u_w_zl))) = -inf;
 
 % optimization choose of aa'
 u_w_zh + bbeta * (
 
% c_r(find(c_r <=0)) = NaN;
% 
% % utility of the retirement
% u_r = (c_r .^ ((1-ssigma) .* ggama))./(1-ssigma);
% u_r(find(isnan(u_r))) = -inf;
end


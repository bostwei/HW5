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

% labor space
% Here I do not use the [0,1] because we  
llb = 0.00000001;
lub = 0.99999999;

L = linspace(llb,lub,Na)';
l = L* ones(1,Na);


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


% Initiate the decision rule for people d_l(a| j) d_aa(a|j)
% - row of each is the a
% - collumn is the age
dec_l_zh = zeros(Na,JR-1);
dec_aa_zh = zeros(Na,JR-1);

dec_l_zl = zeros(Na,JR-1);
dec_aa_zl = zeros(Na,JR-1);


for j = JR-1
% searh for the best l an aa' combination for each a
for i = 1:Na
    ai = A(i);    
    % ------------ the labor l and future asset aaa choice of worker --------------------
    % consumption of worker
    c_w_zh = w * (1-ttheta)* e(1,j)* l + (1+r) * ai - aa';
    c_w_zl = w * (1-ttheta)* e(2,j)* l + (1+r) * ai - aa';
    c_w_zh(find(c_w_zh <0)) = NaN; 
    c_w_zl(find(c_w_zl <0)) = NaN;
  % utility of the worker is 
    u_w_zh = (c_w_zh.^ggama.*(1-l).^(1-ggama)).^(1-ssigma)/(1-ssigma);
    u_w_zl = (c_w_zl.^ggama.*(1-l).^(1-ggama)).^(1-ssigma)/(1-ssigma);
    u_w_zh(find(isnan(u_w_zh))) = -inf;
    u_w_zl(find(isnan(u_w_zl))) = -inf;
    
    w_zh = u_w_zh + bbeta * v0_w_zh(:,j+1)';
    w_zl = u_w_zl + bbeta * v0_w_zl(:,j+1)';
   
   % --------------make choice of (l,aa) given a ------------------
   % we frist choose aa
   % - v1_zh_aa the optimal value w_zh after choosing aa, given a. The row of
   % w_zh is varies of l.
   % - dec_zh_aa is the optimal choice of aa given varies of l
   [v1_zh_aa, dec_zh_aa] = max(w_zh,[],2);
   [v1_zl_aa, dec_zl_aa] = max(w_zl,[],2); 
   
   
   % then choose l
   % - v1_zh_aa the optimal value w_zh after choosing aa, given a. The row of
   % w_zh is varies of l.
   % - dec_zh_aa is the optimal choice of aa given varies of l   
   [v1_zh_l,dec_zh_l] = max(v1_zh_aa);
   [v1_zl_l,dec_zl_l] = max(v1_zl_aa);
   
   % storage the choice of aa, l in the matrix
   % for the good worker
    dec_l_zh(i,j) = dec_zh_l;
    dec_aa_zh(i,j) = dec_zh_aa(dec_zh_l);
    
   % for the bad worker 
    dec_l_zl(i,j) = dec_zl_l;
    dec_aa_zl(i,j) = dec_zl_aa(dec_zl_l);
   
    % storage the value function
    v1_w_zh(i,j) = v1_zh_l;
    v1_w_zl(i,j) = v1_zl_l;
    
    v0_w_zh(:,j) = v1_w_zh(:,j); 
    v0_w_zl(:,j) = v1_w_zl(:,j); 
end
    
end

%------- For the rest of the working agent  -------------------------------
 for j = JR-2:-1:1
     tic
for i = 1:Na
    ai = A(i);    
    % ------------ the labor l and future asset aaa choice of worker --------------------
    % consumption of worker
    c_w_zh = w * (1-ttheta)* e(1,j)* l + (1+r) * ai - aa';
    c_w_zl = w * (1-ttheta)* e(2,j)* l + (1+r) * ai - aa';
    c_w_zh(find(c_w_zh <0)) = NaN; 
    c_w_zl(find(c_w_zl <0)) = NaN;
  % utility of the worker is 
    u_w_zh = (c_w_zh.^ggama.*(1-l).^(1-ggama)).^(1-ssigma)/(1-ssigma);
    u_w_zl = (c_w_zl.^ggama.*(1-l).^(1-ggama)).^(1-ssigma)/(1-ssigma);
    u_w_zh(find(isnan(u_w_zh))) = -inf;
    u_w_zl(find(isnan(u_w_zl))) = -inf;
    
    w_zh = u_w_zh + bbeta * (Pi(1,1)*v0_w_zh(:,j+1)' + Pi(1,2)*v0_w_zl(:,j+1)');
    w_zl = u_w_zl + bbeta * (Pi(2,1)*v0_w_zl(:,j+1)' + Pi(2,2)*v0_w_zl(:,j+1)') ;
   
   % --------------make choice of (l,aa) given a ------------------
   % we frist choose aa
   % - v1_zh_aa the optimal value w_zh after choosing aa, given a. The row of
   % w_zh is varies of l.
   % - dec_zh_aa is the optimal choice of aa given varies of l
   [v1_zh_aa, dec_zh_aa] = max(w_zh,[],2);
   [v1_zl_aa, dec_zl_aa] = max(w_zl,[],2); 
   
   
   % then choose l
   % - v1_zh_aa the optimal value w_zh after choosing aa, given a. The row of
   % w_zh is varies of l.
   % - dec_zh_aa is the optimal choice of aa given varies of l   
   [v1_zh_l,dec_zh_l] = max(v1_zh_aa);
   [v1_zl_l,dec_zl_l] = max(v1_zl_aa);
   
   % storage the choice of aa, l in the matrix
   % for the good worker
    dec_l_zh(i,j) = dec_zh_l;
    dec_aa_zh(i,j) = dec_zh_aa(dec_zh_l);
    
   % for the bad worker 
    dec_l_zl(i,j) = dec_zl_l;
    dec_aa_zl(i,j) = dec_zl_aa(dec_zl_l);
   
    % storage the value function
    v1_w_zh(i,j) = v1_zh_l;
    v1_w_zl(i,j) = v1_zl_l;
    
    v0_w_zh(:,j) = v1_w_zh(:,j); 
    v0_w_zl(:,j) = v1_w_zl(:,j);  
   
end
fprintf('The current age group is %d .\n',j);
toc
 end
% plot the policy function for h0d0
figure(1)
plot(A,A(dec_aa_zh(:,1)),A,A(dec_aa_zl(:,1)));% the policy function for employment state
legend({'high efficiency policy function','low efficiency policy function'},'Location','southeast')
xlabel('a') 
ylabel('aa')
refline(1,0) 
 
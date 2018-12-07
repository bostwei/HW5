%*************************************************************************
% Econ 899 HW5 
% Shiyan Wei
% 11/18/2018
% ************************************************************************

% ************************************************************************
% This script is used to find equilibrium price in OLG Model
%               with theta = 0
%       in the homework 5 for Econ 899
% ***********************************************************************


clear
clc
close all

%parpool('local',16)
%% Question 3 Find the equilibrium price
% Data Initiated 
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
ttheta  = 0;

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

bbeta = 0.97;

% Asset Space
alb = 0.00000001;
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

% initiate the density mu of each age corhort, mu(i) is the density of 
% agent in age i.   
mu = ones(N,1);
for i = 1: N-1
    mu(i+1) = mu(i)/(1+n);
end
% normalized mu, the sized, of population to be 1
mu = mu./sum(mu);



%% Dynamic Programming Problem
% w = 1.05;
% r = 0.05;
% b = 0.2;
% Intital guess of the labor supply
L0 =  0.3132; %sum(sum(mu(1:JR-1)));

% Inital guess of capital 
K0 = 2.1439; %

%-------------- Begin trail and error to find the euqilibirumn wage -----
diff = 10;
Iter = 0;
tol_price = 0.001;

while diff > tol_price
tic
% compute the wage and interest rate given guess L0 and K0
w = (1-aalpha) * (K0 / L0) ^aalpha; 
r = aalpha * (L0 / K0) ^ (1-aalpha) - delta;
% pention benefit
b = (ttheta * w *L0)/(sum(mu(JR:N)));


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
% 
% plot(A,v1_r(:,4))

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
% searh for the best l an aa' combination for each a
v0_w_zh_temp = v0_w_zh(:,j+1);
v0_w_zl_temp = v0_w_zl(:,j+1);
e1j = e(1,j);
e2j = e(2,j);

parfor i = 1:Na
    ai = A(i);    
    % ------------ the labor l and future asset aaa choice of worker --------------------
    % consumption of worker
    c_w_zh = w * (1-ttheta)* e1j * l + (1+r) * ai - aa';
    c_w_zl = w * (1-ttheta)* e2j * l + (1+r) * ai - aa';
    c_w_zh(c_w_zh <0) = NaN; 
    c_w_zl(c_w_zl <0) = NaN;
  % utility of the worker is 
    u_w_zh = (c_w_zh.^ggama.*(1-l).^(1-ggama)).^(1-ssigma)/(1-ssigma);
    u_w_zl = (c_w_zl.^ggama.*(1-l).^(1-ggama)).^(1-ssigma)/(1-ssigma);
    u_w_zh(isnan(u_w_zh)) = -inf;
    u_w_zl(isnan(u_w_zl)) = -inf;
    
    w_zh = u_w_zh + bbeta * v0_w_zh_temp';
    w_zl = u_w_zl + bbeta * v0_w_zl_temp';
   
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
%     fprintf('This is %d th a = %.3f\n',i,ai);
end

    v0_w_zh(:,j) = v1_w_zh(:,j); 
    v0_w_zl(:,j) = v1_w_zl(:,j); 
end

%------- For the rest of the working agent  -------------------------------
 for j = JR-2:-1:1
    v0_w_zh_temp = v0_w_zh(:,j+1);
    v0_w_zl_temp = v0_w_zl(:,j+1);
    e1j = e(1,j);
    e2j = e(2,j);
    P11= Pi(1,1);
    P21= Pi(2,1);
    P12= Pi(1,2);
    P22= Pi(2,2);
parfor i = 1:Na
    ai = A(i);    
    % ------------ the labor l and future asset aaa choice of worker --------------------
    % consumption of worker
    c_w_zh = w * (1-ttheta)* e1j* l + (1+r) * ai - aa';
    c_w_zl = w * (1-ttheta)* e2j* l + (1+r) * ai - aa';
    c_w_zh(c_w_zh <0) = NaN; 
    c_w_zl(c_w_zl <0) = NaN;
  % utility of the worker is 
    u_w_zh = (c_w_zh.^ggama.*(1-l).^(1-ggama)).^(1-ssigma)/(1-ssigma);
    u_w_zl = (c_w_zl.^ggama.*(1-l).^(1-ggama)).^(1-ssigma)/(1-ssigma);
    u_w_zh(isnan(u_w_zh)) = -inf;
    u_w_zl(isnan(u_w_zl)) = -inf;
    
    w_zh = u_w_zh + bbeta * (P11 * v0_w_zh_temp' + P12 * v0_w_zl_temp');
    w_zl = u_w_zl + bbeta * (P21 * v0_w_zh_temp' + P22 * v0_w_zl_temp') ;
   
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
   
end
    v0_w_zh(:,j) = v1_w_zh(:,j); 
    v0_w_zl(:,j) = v1_w_zl(:,j);  
fprintf('The current age group is %d .\n',j);
 end
% % plot the policy function for h0d0
% figure(1)
% plot(A,A(dec_aa_zh(:,1)),A,A(dec_aa_zl(:,1)));% the policy function for employment state
% legend({'high efficiency policy function','low efficiency policy function'},'Location','southeast')
% xlabel('a') 
% ylabel('aa')
% refline(1,0) 

%% Stationary distribution

% % initiate the density mu of each age corhort, mu(i) is the density of 
% % agent in age i.   
% mu = ones(N,1);
% for i = 1: N-1
%     mu(i+1) = mu(i)/(1+n);
% end
% % normalized mu, the sized, of population to be 1
% mu = mu./sum(mu);

% -------- compute the asset choice --------------------
% compute the decision matrix for working agent
% - first argument is a
% - second argument is its asset choice in the next period aa
% - the third argument is age j
g_aa_zh = zeros(Na,Na,JR-1);
g_aa_zl = zeros(Na,Na,JR-1);
for j = 1:JR-1
    for i = 1:Na
        % create the transition criterion
        g_aa_zh(i,dec_aa_zh(i,j),j) = 1;
                % create the transition criterion
        g_aa_zl(i,dec_aa_zl(i,j),j) = 1;
    end
end

% compute the decision matrix for retiring agent
% - first argument is a
% - second argument is its asset choice in the next period aa
% - the third argument is age j

g_aa_r = zeros(Na,Na,N-JR);
% replace the last collumn to always choose asset 1
dec_r(:,N-JR) = ones(Na,1);
for j = 1:N-JR % the reason that I putt J-JR-1 here is that the last coloum of dec_r = 0
    for i = 1:Na
        % create the transition criterion
        g_aa_r(i,dec_r(i,j),j) = 1;
                % create the transition criterion
        g_aa_r(i,dec_r(i,j),j) = 1;
    end
end

% compute the transition matrix for working agent
 trans_aa=zeros(2*Na,2*Na,JR-1);
for j = 1:JR-1
    trans_aa(:,:,j) = [g_aa_zh(:,:,j)*Pi(1,1), g_aa_zl(:,:,j)*Pi(1,2) 
                       g_aa_zh(:,:,j)*Pi(2,1), g_aa_zl(:,:,j)*Pi(2,2) ];
    trans_aa(:,:,j) = trans_aa(:,:,j)'; 
end


% --- compute the labor choices for working agent ----
g_l_zh = zeros(Na,Na,JR-1);
g_l_zl = zeros(Na,Na,JR-1);
for j = 1:JR-1
    for i = 1:Na
        % create the transition criterion
        g_l_zh(i,dec_l_zh(i,j),j) = 1;
                % create the transition criterion
        g_l_zl(i,dec_l_zl(i,j),j) = 1;
    end
end

% -------- the wealth distribution of each cohort -------------
% - phi_zh(a,j) is the density of agent asset holding aa in age j with high
% working efficiency
% - phi_zl(a,j) is the density of agent asset holding aa in age j with low
% working efficiency
phi_zh = zeros(Na,JR-1); 
phi_zl = zeros(Na,JR-1);

% initate the new born generate people will be hold 0 asset
phi_zh(1,1) = Pz(1); 
phi_zl(1,1) = Pz(2); 

phi = [phi_zh;phi_zl];

% calculate the transition of asset holding choice for working agent
for j = 1:JR-1

    phi(:,j+1) = trans_aa(:,:,j) *  phi(:,j);

end

phi_zh = phi(1:Na,:);
phi_zl = phi(Na+1:2*Na,:);

% calcualte the transition of asset holding for retired agent
phi_r = zeros(Na,N-JR);
% extract the asset holding before retire
retire_a = reshape(phi(:,JR),[Na,2]);
% sum over the asset holding choice over
phi_r(:,1) = sum(retire_a,2);

% calculate the transition of asset holding choice for retired agent
% Note: The first period is the same as the last period of working agent 
for j = 1:N-JR

    phi_r(:,j+1) = g_aa_r(:,:,j)' *  phi_r(:,j);

end



% divid population amount into working group and retire group
mu_work = diag(mu(1:JR-1));
mu_r = diag(mu(JR:N));

% rescale the working population density of each age cohort by age amount
mu_phi_zh = phi_zh(:,1:JR-1) * mu_work;
mu_phi_zl = phi_zl(:,1:JR-1) * mu_work;

% rescale the retiring population density of each age cohort by age amount
mu_phi_r = phi_r * mu_r;


%% Updating Criterian

K_new = sum(A.* sum((mu_phi_zh + mu_phi_zl),2) + A.* sum(mu_phi_r,2));

% ---- calculate the labor choices distribution -----------------
% mu_l_zh is the labor choice density after adjust the cohor distribution

mu_l_zh = zeros(Na,JR-1);
mu_l_zl = zeros(Na,JR-1);
for j = 1:JR-1

    mu_l_zh(:,j) = g_l_zh(:,:,j)' *  mu_phi_zh(:,j);
    mu_l_zl(:,j) = g_l_zl(:,:,j)' *  mu_phi_zl(:,j);
end

% calcualte the labor offer for different z
l_zh = L.* mu_l_zh * diag(e(1,:));
l_zl = L.* mu_l_zl * diag(e(2,:));

L_new = sum(sum(l_zh+l_zl));


% update K and L
K1 = 0.9 * K0 + 0.1 * K_new;
L1 = 0.9 * L0 + 0.1 * L_new;

Iter = Iter + 1;

diff = [K_new-K0, L_new-L0];
diff = max(abs(diff));

K0 = K1;
L0 = L1;
toc

fprintf('Iteration %d: wage difference is %.4f, wage is %.4f, interest rate is %.4f \n ', Iter, diff,w, r);


end

%% Total Welfare sum of the v
w_working = v1_w_zh(:,1:45) .*  mu_phi_zh + v1_w_zl(:,1:45) .* mu_phi_zl;
w_working = sum(sum(abs(w_working)));
w_retire = v1_r.* mu_phi_r(:,1:20);
w_retire = sum(sum(abs(w_retire)));
welfare  = w_retire + w_working;

%% Total wealth sum of a across population
A_working = A .*  mu_phi_zh + A .*mu_phi_zl; 
A_retire = A .* mu_phi_r;
CV = sum(sum(A_working)) + sum(sum(A_retire));

%% Consumption Equivalance 
% ss0 is the value function without social security.
v_zh_ss0 = v1_w_zh;
v_zl_ss0 = v1_w_zl;
v_retire_ss0 = v1_r;
save('ss0.mat','v_zh_ss0','v_zl_ss0','v_retire_ss0');
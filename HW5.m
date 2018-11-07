%*************************************************************************
% Econ 899 HW5 
% Shiyan Wei
% 11/06/2018
% ************************************************************************

% ************************************************************************
% This script is used to solving the homework 5 for Econ 899
% ***********************************************************************

%% Data Initiated 
N = 66;
n = 0.011;
% Grid Number
M = 200;

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

% Trhansition probability
Pi = [0.9261 1-0.9261;
     1-0.9811 0.9811];

% capital share
aalpha = 0.36;
% depreciation rate
delta = 0.06;

% Asset Space
alb = -2;
aub = 5;



A = linspace(alb, aub, M)';

a = A* ones(1,M);
aa = A* ones(1,M); 

%% Dynamic Programming Problem
w = 1.05;
r = 0.05;
b = 0.2;


%------------- The utility for the retireed agent ----------------
% consumption of returement
 
c_r = (1+r)*a + b - aa';
c_r(find(c_r <=0)) = NaN;




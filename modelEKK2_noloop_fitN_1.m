%
%
%   Evaluate our fit to the moments
%     (based on modelEKK)
%
%    *** using the real data ***
%    using RnF to back out M_n + B_n
%    parameterize lambda_ni based on pi_ni
%

% Only the Parameters sections should be modified.

% Parameters - Tolerance 
Mntol = .000001;    % Tolerance for S iteration
maxitMn = 3000;          % Max. number of iterations for wages
maxitMn = 1000;
vfactorS = .1;           % v factor for Alvarez - Lucas algorithm
vfactorS = 1;

% dimensional settings
Ncountry = 25;
NB = 150;
iFR = 10;
iDE = 5;

% Initial guess for Parameters

lambda_vec_FR = ones(Ncountry,1);
%lambda_vec_FR(1:9) = [0.1031 0.6165 0.7144 0.5999 0.2760 0.1781 1.1379 0.3869 0.2121];
%lambda_vec_FR(1:9) = [ 0.1283 0.5994 0.5120 0.5066 0.2544 0.1797 0.9998 0.6077 0.2736];
lambda_vec_FR(1:9) = [0.2447 1.3687 0.2108 0.4783 0.8070 0.2776 1.1050 0.2623 0.2249];

%lambda_vec_FR(11:25) = [0.4731 0.6616 0.0753 0.4074 0.8713 1.1091 0.7899 0.7122 0.0980 0.2580 1.1936 1.0006 0.0509 0.1163 0.3370];
%lambda_vec_FR(11:25) = [ 0.4906 0.6356 0.0770 0.3939 0.7521 0.6791 0.7493 0.5692 0.0907 0.3108 1.1835 0.8829 0.0574 0.1440 0.3582];
lambda_vec_FR(11:25) = [0.7180 0.4116 0.6293 0.2999 0.6800 0.2644 0.3420 0.2635 0.7002 0.4401 0.7079 0.4816 0.4196 0.5662 0.3528];

theta = 4; % parameter of heterogeneity in firm-level efficiency
%theta = parmvec(1);
theta = 2.8842;
theta = 3.2146;

phi = 2; % parameter of task-firm-specific idiosyncratic labor efficiency
%phi = parmvec(1); 
phi = 2.1849;
phi = 1.9521;

funnyphi = 0.2; % congestion
%funnyphi = parmvec(1);
funnyphi = 0.2476;
funnyphi = 0.1873;

psi = 0.3; % value of off-diagonal bilaterl lambdas
%psi = parmvec(2);
psi = 0.5250;

lambda1 = .5; % outsourcing rate for blue-collar tasks
%lambda1 = parmvec(3);
lambda1 = 0.3827;
lambda1 = 0.2071;

beta0 = 0.4; % share of purchased services
%beta0 = parmvec(1);
beta0 = 0.3;

K = 15;  % number of blue-collar tasks
%K = parmvec(1);
K = 20;

% load the moments that we condition on

% bilateral trade share matrix

data1 = csvread('Dataset1_only.csv');
pimat = vec2mat(data1,Ncountry);
pimat = pimat';  % exporters are columns, columns sum to less than 1 since no ROW


data1_bi = csvread('Dataset1_Bitrade_only.csv');
trademat = vec2mat(data1_bi(:,1),Ncountry);  % levels of bilateral trade
trademat = trademat';

ypartial = sum(trademat);  % now its a row vector

% population and value added share

data2 = csvread('Dataset2_only.csv');

Lvec = data2(:,1);
Lvec = Lvec';
Lvec = Lvec / 1000;  % express population in thousands

LshareMvec = data2(:,2);
LshareMvec = LshareMvec';

Lvec

LshareMvec

% data on French firms (for now just need Relationships)

data3 = csvread('Dataset3_only.csv');

RnFvec = data3(:,2); % relationships involving France as source
RnFvec = RnFvec';
RnFvec = RnFvec / 1000; % express relationships, and other counts, in thousands

data1_FR = csvread('Dataset1_France_only.csv');
absorp = data1_FR;  % absorption

% construct bilateral lambdas

lambda_ni = ones(Ncountry,Ncountry);
lambda_ni(:,iFR) = lambda_vec_FR;

% Normalize Svec so that it sums to 1 (Svec should be irrelevant)        
Svec = ypartial .^ (1.3);
Svec = Svec / sum(Svec);
   
Upsvec = Svec ./ (diag(pimat)');
   
homeshares = diag(pimat);
 
taumat = (Svec' ./ Svec) .* pimat ./ (homeshares * ones(1,Ncountry));
   
dnimat_mtheta = taumat ./ lambda_ni;

dnimat = dnimat_mtheta .^ (-1/theta);
   
Xi1vec = (lambda1 * theta / phi) * ((1 - beta0) ./ (1 - beta0 - LshareMvec)) .* (Lvec .^ (-funnyphi)) .* (Upsvec .^ (phi / theta));   

% measure of buyers (manufacturing firms + others)
Mvec_Bvec = (1 / K) * ((1 - beta0) ./ (1 - beta0 - LshareMvec)) .* (RnFvec ./ pimat(:,iFR)');
   
eta = @(n,i,c) ...
    lambda_ni(n,i) * Mvec_Bvec(n) * Lvec(n)^(- funnyphi) * (Upsvec(n) * c ^ theta) ^ ((phi/theta) - 1) ...
      * lambda1 * K * exp( - Xi1vec(n) * c ^ phi);
    
etaWi = @(i,c) ... 
    sum(arrayfun(eta,1:Ncountry,i * ones(1,Ncountry),c * dnimat(:,i)'));
    
etaWvec = @(c) ...
    arrayfun(etaWi,1:Ncountry,c * ones(1,Ncountry));
    
integrandW = @(c) (ones(1,Ncountry) - exp( -etaWvec(c) ) ) * theta * c ^ (theta - 1);

Mvec = integral(integrandW,0,Inf,'ArrayValued',true) .* Svec ;
    
% Prediction for relationships (should fit perfectly)
RvecFR = K * ((1 - beta0 - LshareMvec) ./ (1 - beta0)) .* pimat(:,iFR)' .* Mvec_Bvec;

disp('relationships, model and data');
disp([RvecFR' RnFvec']');

disp('Mvec, Mvec_Bvec');
disp([Mvec' Mvec_Bvec']');


% The following are the Data moments we'll try to fit

D_expectedD = data3(:,8);
D_expectedD(iFR) = 1;

D_R = data3(:,2) / 1000;
D_R(iFR) = 1;

D_N = data3(:,1) / 1000;
D_N(iFR) = 1;

D_R_over_N_FR = D_R ./ D_N;

D_NcDE = data3(:,3) / 1000;
D_NcDE(iFR) = 1;

D_NcDE_over_N_FR = D_NcDE ./ D_N;

D_RcDE_over_NcDE_FR = data3(:,4);  % Francis calculated ratio
D_RcDE_over_NcDE_FR(iFR) = 1; 

D_RcDE = D_NcDE .* D_RcDE_over_NcDE_FR; 

D_R_over_N_FR = D_R ./ D_N;


D_perc50 = data3(:,5);
D_perc50(iFR) = 1; 

D_perc90 = data3(:,6);
D_perc90(iFR) = 1;

D_perc99 = data3(:,7);
D_perc99(iFR) = 1;
 
% Now back to computing Model moments from this run

% Expected # iFR suppliers per buyer in n (with at least 1)

outvec = (1 - beta0 - LshareMvec) ./ (1 - beta0); % outsourcing probability by destination
 
prob0 = (1 - pimat(:,iFR)' .* outvec) .^ K; % prob of no iFR suppliers

M_expectedD = K * pimat(:,iFR)' .* outvec ./ (1 - prob0);
M_expectedD(iFR) = 1;

M_expectedD = M_expectedD';

% Now calculate entry by French firms, and relationships in Germany conditional on entry
   
   etaveci = @(i,c) ...
       arrayfun(eta,1:Ncountry,i * ones(1,Ncountry),c * ones(1,Ncountry));
    
   etavecdi = @(i,c) ...
       arrayfun(eta,1:Ncountry,i * ones(1,Ncountry),c * dnimat(:,i)');
   
   integrand = @(c) ...
       (ones(1,Ncountry) - exp( -etaveci(iFR,c) ) ) * theta * c ^ (theta - 1);
   
   NvecFR = Svec(iFR) * integral(integrand,0,Inf,'ArrayValued',true) .* (dnimat(:,iFR) .^ (-theta))';
     
   integrand = @(c) (1 - exp( -eta(iDE,iFR,c * dnimat(iDE,iFR)) ) ) * ...
       (ones(1,Ncountry) - exp( -etavecdi(iFR,c) ) ) * theta * c ^ (theta - 1);
   
   NveccDE = Svec(iFR) * integral(integrand,0,Inf,'ArrayValued',true);
   
   integrand = @(c) eta(iDE,iFR,c * dnimat(iDE,iFR)) * (1 - exp( -eta(iDE,iFR,c * dnimat(iDE,iFR)) ) ) * ...
       (ones(1,Ncountry) - exp( -etavecdi(iFR,c) ) ) * theta * c ^ (theta - 1);
   
   RveccDE = Svec(iFR) * integral(integrand,0,Inf,'ArrayValued',true);
 
% Conditioning shouldn't matter in iDE

NveccDE(iDE) = NvecFR(iDE);
RveccDE(iDE) = RnFvec(iDE);
   
% Source country playing the role of France

% Calculate distribution of # buyers for firms from chosen source
ProbSn_integrand = @(s,n,c) ...
    exp( s * log( eta(n,iFR,c)) - eta(n,iFR,c) - sum(log([1:s]') ) ) * theta * c ^ (theta -1);
Prob_Sn_sn = @(s,n) integral(@(c) ProbSn_integrand(s,n,c),0,Inf,'ArrayValued',true) ...
    * Svec(iFR) * (dnimat(n,iFR) ^ (-theta)) / (NvecFR(n));

Prob_Sn = arrayfun(Prob_Sn_sn,[1:NB]' * ones(1,Ncountry),ones(NB,1) * [1:Ncountry]);

% Probability function of S_n for each destination country 
% (first column represents S_n = s and each other column represents a
% country.
%[[1:NB]' Prob_Sn ]

% CDF of S_n for each destination country
%  with iFR always the source country
CDF_Sn = tril(ones(NB,NB)) * Prob_Sn ;
%[[1:NB]' CDF_Sn]


NvecFR(iFR) = 1;

NveccDE(iFR) = 1;

RveccDE(iFR) = 1;

RvecFR(iFR) = 1;

M_R_over_N_FR = RvecFR' ./ NvecFR';

M_NcDE_over_N_FR = NveccDE' ./ NvecFR';

M_RcDE_over_NcDE_FR = RveccDE' ./ NveccDE';

M_perc50 = sum(CDF_Sn < .50)' + ones(Ncountry,1);
M_perc75 = sum(CDF_Sn < .75)' + ones(Ncountry,1);
M_perc90 = sum(CDF_Sn < .90)' + ones(Ncountry,1);
M_perc99 = sum(CDF_Sn < .99)' + ones(Ncountry,1);

M_perc50(iFR) = 1;
M_perc50(iFR) = 1;
M_perc90(iFR) = 1;
M_perc99(iFR) = 1;


disp('Relationships, model and data');
disp([RvecFR' D_R]');

disp('French exporters, model and data');
disp([NvecFR' D_N]');

disp('expected suppliers per buyer, model and data');
disp([M_expectedD D_expectedD]');

disp('Relationships conditional, model and data');
disp([RveccDE' D_RcDE]');

disp('French exporters conditional, model and data');
disp([NveccDE' D_NcDE]');

disp('relationships per French exporter, model and data');
disp([M_R_over_N_FR D_R_over_N_FR]');

disp('French exporters (and to Germany) per French exporter, model and data');
disp([M_NcDE_over_N_FR D_NcDE_over_N_FR]');

disp('relationships in German per French exporter also selling elsewhere, model and data');
disp([M_RcDE_over_NcDE_FR D_RcDE_over_NcDE_FR]');

disp('median buyers per French exporter, model and data');
disp([M_perc50 D_perc50]');

disp('75th percentile of buyers per French exporter, model');
disp([M_perc75]');

disp('90th percentile of buyers per French exporter, model and data');
disp([M_perc90 D_perc90]');

disp('99th percentile of buyers per French exporter, model and data');
disp([M_perc99 D_perc99]');

%taumat
%dnimat
%lambda_ni

% remove  M_expectedD from objective function, and fix K in the estimation
% + sum((M_expectedD - D_expectedD).^2,1)

loss = sum((RvecFR' - D_R).^2,1) + sum((NvecFR' - D_N).^2,1) ...
    + sum((M_perc50 - D_perc50).^2,1) ...
    + sum((M_perc90 - D_perc90).^2,1) + sum((M_perc99 - D_perc99).^2,1) ...
    + sum((RveccDE' - D_RcDE).^2,1) ...
    + sum((NveccDE' - D_NcDE).^2,1);
loss




%display('Relationships regressed on market size')
%y = log(Rvec)';
%X = [ones(Ncountry,1) log(mfg_grossprod)'];
%[b, bint] = regress(y,X);
%b

%display('Entry regressed on market size')
%y = log(Nvec)';
%[b, bint] = regress(y,X);
%b

noFR = [1:(iFR-1) (iFR+1):Ncountry];

xvar = D_NcDE(noFR);
yvar = D_RcDE_over_NcDE_FR(noFR);

disp('Data: relationships in DE for FR firms (given buyers in n) regressed on # with buyers in n')
yFR = log(yvar);
X1FR = [ones((Ncountry-1),1) log(xvar)];
[b, bint] = regress(yFR,X1FR);
b


xvar = NveccDE(noFR)';
yvar = M_RcDE_over_NcDE_FR(noFR);

disp('Model: relationships in DE for FR firms (given buyers in n) regressed on # with buyers in n')
yFR = log(yvar);
X1FR = [ones((Ncountry-1),1) log(xvar)];
[b, bint] = regress(yFR,X1FR);
b


share_FR = pimat(:,iFR); 

xvar1 = absorp(noFR);
xvar1a = Mvec_Bvec(noFR)';
xvar2 = share_FR(noFR);


yvar = D_R(noFR);

disp('Data: relationships regressed on mkt size and mkt share')
yFR = log(yvar);
X2FR = [ones((Ncountry-1),1) log(xvar1) log(xvar2)];
[b, bint] = regress(yFR,X2FR);
b

yvar = RvecFR(noFR)';

disp('Model: relationships regressed on mkt size and mkt share')
yFR = log(yvar);
X2FR = [ones((Ncountry-1),1) log(xvar1) log(xvar2)];
[b, bint] = regress(yFR,X2FR);
b

yvar = D_N(noFR);

disp('Data: exporters regressed on mkt size and mkt share')
yFR = log(yvar);
X2FR = [ones((Ncountry-1),1) log(xvar1) log(xvar2)];
[b, bint] = regress(yFR,X2FR);
b

yvar = NvecFR(noFR)';

disp('Model: exporters regressed on mkt size and mkt share')
yFR = log(yvar);
X2FR = [ones((Ncountry-1),1) log(xvar1) log(xvar2)];
[b, bint] = regress(yFR,X2FR);
b

%
% *** Figures ***
%

loglog(D_R(noFR),RvecFR(noFR)','kx');
hold on;
title('Figure 1: Relationships,  Model and Data');
xlabel('data');
ylabel('model');
hold off;
pause


loglog(D_N(noFR),NvecFR(noFR)','kx');
hold on;
title('Figure 2: Exporters, Model and Data');
xlabel('data');
ylabel('model');
hold off;
pause


loglog(D_perc99(noFR),M_perc99(noFR)','kx');
hold on;
title('Figure 4: Buyers per French exporter (99th perc.), Model and Data');
xlabel('data');
ylabel('model');
hold off;
pause


h = loglog(absorp(noFR),(NvecFR(noFR)' ./ share_FR(noFR)),'kx');
hold on;
title('Figure 2: French Entrants (adjusted for mkt share) and Market Size');
xlabel('manufacturing absorption');
ylabel('number of French entrants (adjusted for Fr mkt shr), by destination');
hold off;
pause


i = loglog(absorp(noFR),M_R_over_N_FR(noFR)','kx');
hold on;
title('Figure 3: Buyers per French Exporter');
xlabel('manufacturing absorption');
ylabel('buyers per supplier, by destination');
hold off; 
pause


j = loglog(absorp(noFR),(RvecFR(noFR)' ./ share_FR(noFR)),'kx');
hold on;
title('Figure 4: Relationships of French Exporters (adjusted for mkt shr) and Market Size');
xlabel('manufacturing absorption');
ylabel('relationships of French entrants (adjusted for Fr mkt shr), by destination');
hold off;
pause

%%

figure(1)
loglog(D_NcDE(noFR),D_RcDE_over_NcDE_FR(noFR),'kx', ...
    NveccDE(noFR)',M_RcDE_over_NcDE_FR(noFR),'rs');
title('Customers in Germany per French Exporter, Conditional')
xlabel('number exporting to Germany and elsewhere');
ylabel('customers in Germany per exporter, conditional');
legend('Data','Model');
pause

% Belgium
BE_raw = csvread('Michael_BE.csv');
D_NcBE = BE_raw(:,2) / 1000;
D_NcBE(iFR) = 1;
D_RcBE_over_NcBE_FR = BE_raw(:,1);
D_RcBE_over_NcBE_FR(iFR) = 1;

iBE = 2;

integrand = @(c) (1 - exp( -eta(iBE,iFR,c * dnimat(iBE,iFR)) ) ) * ...
       (ones(1,Ncountry) - exp( -etavecdi(iFR,c) ) ) * theta * c ^ (theta - 1);
   
   NveccBE = Svec(iFR) * integral(integrand,0,Inf,'ArrayValued',true);
   
   integrand = @(c) eta(iBE,iFR,c * dnimat(iBE,iFR)) * (1 - exp( -eta(iBE,iFR,c * dnimat(iBE,iFR)) ) ) * ...
       (ones(1,Ncountry) - exp( -etavecdi(iFR,c) ) ) * theta * c ^ (theta - 1);
   
   RveccBE = Svec(iFR) * integral(integrand,0,Inf,'ArrayValued',true);
 
% Conditioning shouldn't matter in iDE

NveccBE(iBE) = NvecFR(iBE);
RveccBE(iBE) = RnFvec(iBE);

NveccBE(iFR) = 1;
RveccBE(iFR) = 1;

M_NcBE_over_N_FR = NveccBE' ./ NvecFR';
M_RcBE_over_NcBE_FR = RveccBE' ./ NveccBE';

l = loglog(D_NcBE(noFR),D_RcBE_over_NcBE_FR(noFR),'kx', ...
    NveccBE(noFR)',M_RcBE_over_NcBE_FR(noFR),'rs');
title('Customers in Belgium per French Exporter, Conditional')
xlabel('number exporting to Belgium and elsewhere');
ylabel('customers in Belgium per exporter, conditional');
legend('Data','Model');

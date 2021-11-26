%  MASWaves Inversion
%  Version: 06.2020
%%
%  Test_MASWaves_Inversion_MC
%  
%% Load and plot sample data
SampleData = readtable('SampleData_MASWavesInversion.csv');
c_OBS = SampleData{1:end,1};
up_low_boundary = 'yes'; % Upper/lower boundary curves are available
c_OBS_low = SampleData{1:end,2};
c_OBS_up = SampleData{1:end,3};
lambda_OBS = SampleData{1:end,4};

figure, hold on
exp_curve = plot(c_OBS,lambda_OBS,'kx-','MarkerFaceColor',...
    'k','MarkerSize', 2,'LineWidth',1);
exp_low = plot(c_OBS_low,lambda_OBS,'r--','LineWidth',1);
exp_up = plot(c_OBS_up,lambda_OBS,'r--','LineWidth',1);

set(gca,'FontSize',9),axis ij, grid on, box off
xlim([80 200]), ylim([0 35])
xlabel('Rayleigh wave velocity [m/s]','FontSize',9,'Fontweight','normal')
ylabel('Wavelength [m]','FontSize',9,'Fontweight','normal')
legend([exp_curve,exp_low],'mean','mean \pm std','Location','SouthWest')

set(gcf,'units','centimeters')
pos = [2, 2, 8, 10];
set(gcf,'Position',pos)

%% Initialization

% Testing phase velocity
c_min = 75; % m/s
c_max = 300; % m/s
c_step = 0.09; % m/s
delta_c = 2; % m/s

% Initial values for model parameters
n = 4;
h_initial = [2 2 4 6]; % m
beta_initial = [c_OBS(1)*1.09 c_OBS(10)*1.09 c_OBS(17)*1.09...
    c_OBS(23)*1.09 c_OBS(end)*1.09]; % m/s
n_unsat = 1;
nu_unsat = 0.3;
alpha_temp = sqrt((2*(1-nu_unsat))/(1-2*nu_unsat))*beta_initial; % m/s
alpha_initial = [alpha_temp(1) 1440 1440 1440 1440]; % m/s
rho = [1850 1900 1950 1950 1950]; % kg/m^3

%% Run inversion analysis (one initation)

% Number of velocity reversals 
N_reversals = 0; % Normally dispersive analysis

% Search-control parameters
b_S = 5; 
b_h = 10; 
N_max = 1000; 
e_max = 0;
tic
% Run inversion
[store_all,elapsedTime,store_accepted] = MASWaves_inversion_MC...
    (c_min,c_max,c_step,delta_c,n,n_unsat,alpha_initial,nu_unsat,...
    beta_initial,rho,h_initial,N_reversals,c_OBS,lambda_OBS,...
    up_low_boundary,c_OBS_up,c_OBS_low,b_S,b_h,N_max,e_max);
toc
%% Display inversion results 
MaxDepth = 16;

MASWaves_inversion_MC_plot(c_OBS,lambda_OBS,up_low_boundary,c_OBS_up,...
    c_OBS_low,n,store_all,store_accepted,MaxDepth);
 
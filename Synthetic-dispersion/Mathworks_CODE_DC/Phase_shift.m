% MASW_DC
% 
% Calculates the seismic dispersion curve commonly used in the MASW method.
% Here, it's cacluated in two ways, using 1) the Phase-Shift 
% method by Park et al., 1998a and 2) using a 1st order polyfit to the 
% unwrapped phase angle vs position. 
% 
% Links to resources:
% http://www.masw.com/files/DispersionImaingScheme-1.pdf
% http://www.masw.com/DCImagingScheme.html
% http://masw.com/files/RYD-03-02.pdf
%
% This example uses synthetic wavefrom data created using DISECA MATLAB
% script
% http://www.sciencedirect.com/science/article/pii/S0266352X11000425
%
% John Schuh 21 Nov 2017

clear; close all
load SampleData.mat % Load the synthetic dataset
S = load('SampleData.mat');
U=suma_sg; % my x-t domain signals defined for the below script

ct=300:1:1200; % A vector for the velocity of interest in m/s.
               % This is the velocity "scanning" range and was chosen
               % based on the DISECA dispersion example that created the
               % synthetic dataset
               
freqlimits=[15 80]; % This sets the frequency(Hz)plotting range for 
                    % the plot at the end

% These properties of the signal cannont be chagned for the given
% synthetic dataset provided by DISECA
fs=1000; % sampling frequency Hz
t=1/fs:1/fs:length(U(:,1))/fs; % time vector in seconds
min_x=1;                             % min offset [m] of reciever spread
max_x = 100;                         % max offset [m] of reciever spread
d_x = 2;                             % distance between receivers [m] 
x = min_x:d_x:max_x;                 % spatial sampling vector (receiver position)
N = length(U(1,:)); % number of traces

%% Plot seismogram
signal_plot = zeros(length(U(:,1)),N);
for j = 1:N
    signal_plot(:,j) = U(:,j)/50 + (j-1)*d_x;
    plot(signal_plot(:,j),t,'k');
    hold on
end
set(gca,'YDir','reverse')
set(gca, 'FontSize', 12)
ylim([0,1]); xlim([min_x,max_x])
%% Plot signal image
figure;
imagesc(x,t,U) %plot the data in x-t domain (i.e., a shot gather)
title('Shot Gather of Dispersive Waveform (x-t domain)')
xlabel('Position (m)')
ylabel('time (sec)')
set(gca,'FontSize',10)
%% Main part of computation
%%%%% Find the dispersion image %%%%%%
% 1. First, find the fft of each trace
f=linspace(0,fs,length(t)); % freq vector
f=f(1:floor(length(f)/2)); % half the symmetric frequencies
Ufft=fft(U); % fft is perfrom on each column (the time vectorn)
Ufft=Ufft(1:floor(length(Ufft)/2),:); % half the symmetric fft
w=f*2*pi; % convert the freq vector to rads/sec

% 2. Normalize the spectrum
Rnorm=Ufft./abs(Ufft); % Normalize the spectrum
                            
% 3. Perfrom the summation over N in the frequency domain
AsSum=zeros(length(f),length(ct)); % Initialize the dispersion image matrix
for ii=1:length(w) % Iterate over frequency
    [FREQ,junk]=meshgrid(Rnorm(ii,:),ct); % A matrix of a single normalized spectrum term
    [X,Y]=meshgrid(x,ct); % A matrix of positions and velocites
    As=exp(1j*w(ii)*X./Y).*FREQ; % As(ct)
    AsSum(ii,:)=abs(sum(rot90(As)));   % Sum up As for a given frequency. 
end                                    % The final result will be a matrix of 
                                       % amplitudes changing with phase 
                                       % velocity in the x direction and 
                                       % frequency in the y direction                                 
                                 
%%% Solve for the phase velocity directly using the 
%%% slope of phase angle vs position
AngUfft=angle(Ufft); % the phase angle of each spectrum term
UnAngUfft=abs(unwrap(AngUfft')); % phase angle corrected and trasposed
pfk=zeros(1,length(f)); % Initialize the phase velocity vector
for ii=1:length(f) % iterate over the number of frequencies
    pf=polyfit(x',UnAngUfft(:,ii),1); % take the best linear fit to the phase slope
    pfk(ii)=pf(1); % just the slope of phase change
end

phasevel=1./(pfk./(2*pi*f)); % convert the phase slope to a velocity
%% Plot everything
figure;
imagesc(ct,f,AsSum) % plot the dispersion curve image (freq on y vel on x)
colormap(jet)
% view(-90,90) % rotate plot so velocity is on the vertical axis
hold on
plot(smooth(smooth(phasevel,'rlowess')),f,'-ok','linewidth',2) 
% plot a smoothed version of the directly calculated phase vel ('rlowess' works well for data I've used)
% Smooth > moving average filter

%plot(phasevel,f,'k:','linewidth',2) 
% plot the directly calculated phase vel (this can be messy for real data)
ylim(freqlimits)
xlabel('Phase Velocity (m/s)')
ylabel('Freq (Hz)')
title('Dispersion Image')
set(gca,'FontSize',10)


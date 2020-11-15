%% Generate a signal consisting of 3 sinusoids
% one at 50 Hz, one at 120 Hz with twice the amplitude and one at 300Hz.

% Time
sampling_rate = 1/1000; % 1KHz
t = (0:sampling_rate:1)';  
L = length(t) / 2;

% Signal
y = sin(2*pi*50*t) + 2*cos(2*pi*120*t) + sin(2*pi*300*t);
figure; subplot(2,2,1); plot(t(1:100),y(1:100),'LineWidth',3);
grid on; title(sprintf('50Hz sine + 120Hz cos \n + 300Hz sine'),'Fontsize',14);
xlabel('time','Fontsize',12); ylabel('Amplitude','Fontsize',12)

% decompose/reconstruct the signal
coef = fft(y);
subplot(2,2,2); plot(real(coef(1:L)),'LineWidth',3); hold on
plot(imag(coef(1:L)),'r','LineWidth',3); grid on; 
title(sprintf('Real and imaginary part \n of the fft'),'Fontsize',14);
xlabel('frequency','Fontsize',12); ylabel('Amplitude','Fontsize',12)

reconsty = ifft(coef);
subplot(2,2,3); plot(t(1:100),reconsty(1:100),'g','LineWidth',3);
grid on; title('inverse fft','Fontsize',14)
xlabel('time','Fontsize',12); ylabel('Amplitude','Fontsize',12)

diff = y - reconsty;
subplot(2,2,4); plot(t(1:100),diff(1:100),'k','LineWidth',3);
grid on; title('Difference','Fontsize',14)
xlabel('time','Fontsize',12); ylabel('Amplitude','Fontsize',12)

%% Amplitude spectrum

coef = fft(y) / length(t); % normalize by the number of time points
Nyquist_Limit = (1/sampling_rate)/2; % = 500Hz
x = linspace(0,1,length(t)/2)*Nyquist_Limit;
figure; subplot(1,2,1); 
plot(x,abs(coef(1:length(t)/2)),'LineWidth',3);
xlabel('frequency','Fontsize',12); ylabel('amplitude','Fontsize',12)
axis tight; grid on; title('Original Signal')

% imagine we recorded this signal at 200Hz
ys = downsample(y,5); 
Nyquist_Limit = ((1/sampling_rate)/5)/2; % = 100Hz
x = linspace(0,1,length(ys)/2)*Nyquist_Limit;
subplot(1,2,2); coef = fft(ys) / length(ys);
plot(x,abs(coef(1:length(ys)/2)),'LineWidth',3);
xlabel('frequency','Fontsize',12); ylabel('amplitude','Fontsize',12)
axis tight; grid on; title('Sampled at 200Hz')

%% Power

% recompute on the original signal
coef = fft(y) / length(t);
Nyquist_Limit = (1/sampling_rate)/2; % = 500Hz
x = linspace(0,1,L)*Nyquist_Limit;
subplot(1,2,1); plot(x,abs(coef(1:L)),'LineWidth',3);
xlabel('frequency','Fontsize',12); ylabel('amplitude','Fontsize',12)
axis tight; grid on; title('amplitude spectrum')
% get the power
coef = fft(y);
P = coef(1:L).*conj(coef(1:L))/ length(t);
subplot(1,2,2); plot(x,P,'LineWidth',3);
xlabel('frequency','Fontsize',12); ylabel('Power','Fontsize',12)
axis tight; grid on; title('Power spectrum')

%% Average power
hp = spectrum.periodogram('hamming');  % Create periodogram object
hpopts = psdopts(hp,x); % Create options object 
set(hpopts,'Fs',sampling_rate,'SpectrumType','twosided','centerdc',true); % set properties
msspectrum(hp,y,hpopts); % does the same as above automatically

hpsd = psd(hp,y,hpopts); % measures power per unit of frequency
figure; plot(hpsd);

power_freqdomain = avgpower(hpsd);
power_timedomain = sum(abs(y).^2)/length(y);

%% Phase
figure; subplot(1,2,1);
%y = sin(2*pi*50*t) + 2*cos(2*pi*120*t) + sin(2*pi*300*t);
y1 = sin(2*pi*50*t);
y2 = 2*cos(2*pi*120*t);
y3 = sin(2*pi*300*t);
plot(t(1:60),y1(1:60),'LineWidth',3); hold on
plot(t(1:60),y3(1:60),'g','LineWidth',3);
plot(t(1:60),y2(1:60),'r','LineWidth',3); 
grid on; title('Original data','Fontsize',14);
xlabel('time','Fontsize',12); ylabel('Amplitude','Fontsize',12)

theta = atan(imag(coef)./real(coef)); 
subplot(1,2,2); plot(x(1:60),theta(1:60)./(pi/180),'LineWidth',3);
grid on; title('Phase','Fontsize',14); axis tight
xlabel('time','Fontsize',12); ylabel('Phase in degree','Fontsize',12)

% signal processing as a function to do just that
phi = unwrap(angle(coef));  % Phase
figure;plot(x(1:60),phi(1:60)./(pi/180));

%% coherence

[C13,F13] = mscohere(y1,y3,hanning(round(L)),round(L/2),round(L),1/sampling_rate);
figure; plot(F13,C13,'r','LineWidth',3); 
title('Coherence at each frequencies','Fontsize',14); axis tight
xlabel('frequencies','Fontsize',12); ylabel('Coherence','Fontsize',12)
[C12,F12] = mscohere(y1,y2,hanning(round(L)),round(L/2),round(L),1/sampling_rate);
hold on; plot(F12,C12,'LineWidth',3); grid on

%% Time - frequency analysis
clear all
[data,sampling_rate]=wavread('song1.wav');
t = [1:length(data)];
L = length(t) / 2;
 
% Signal
figure; subplot(2,2,1); plot(data,'LineWidth',3);
grid on; title(sprintf('bird song'),'Fontsize',14);
xlabel('time','Fontsize',12); ylabel('Amplitude','Fontsize',12); axis tight
 
% what if we do an fft
coef = fft(data);
P = coef(1:L).*conj(coef(1:L))/ length(t);
Nyquist_Limit = (1/sampling_rate)/2; 
x = linspace(0,1,length(t)/2)*Nyquist_Limit;
subplot(2,2,2); plot(x,P,'LineWidth',3);
xlabel('frequency','Fontsize',12); ylabel('Power','Fontsize',12); axis tight
axis tight; grid on; title('Power spectrum')
 
[S,F,T,P]=spectrogram(data,256,[],[],sampling_rate);
subplot(2,2,4); mesh(abs(P)); 
xlabel('Time (Seconds)'); ylabel('Hz'); zlabel('power')
title('Power Spectral density'); 
set(gca,'YTick',F(1:20:end)); set(gca,'XTick',T(1:300:end))

% time-frequency decomposition
subplot(2,2,3); spectrogram(data,256,[],[],sampling_rate,'yaxis')
title('Spectrogram'); set(gca,'YTick',F(1:20:end)); set(gca,'XTick',T(1:300:end))

% we can also look at the effect of windowing
figure; subplot(4,1,1); w = rectwin(256); % rectangle
spectrogram(data,w,[],[],sampling_rate,'yaxis'); title('rectangular window')

subplot(4,1,2); w = bartlett(256); % like a triangle
spectrogram(data,w,[],[],sampling_rate,'yaxis'); title('bartlett window')

subplot(4,1,3); w = gaussian(256); % gaussian
spectrogram(data,w,[],[],sampling_rate,'yaxis'); title('gaussian window')

subplot(4,1,4); w = hamming(256); % like gaussian - default
spectrogram(data,w,[],[],sampling_rate,'yaxis'); title('hamming window')

% we can also illustrate the time/frequency trade-off
% if we use long windows we can see well low-frequencies
figure
windows = [256, 2048, 8192];
for w=1:3
    subplot(1,3,w); spectrogram(data,windows(w),[],[],sampling_rate,'yaxis')
    title(['Spectrogram with ' num2str((windows(w)/sampling_rate)*1000) 'ms window'])
    [S,F,T,P]=spectrogram(data,256,[],[],sampling_rate);
    set(gca,'YTick',F(1:20:end)); set(gca,'XTick',T(1:300:end))
end


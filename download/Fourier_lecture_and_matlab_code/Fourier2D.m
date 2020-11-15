% Fourier 2D - code to illustrate image decomposition
clear all

% import with imread
im=double(imread('Roberto.jpg'));

% note the last dimention is for the RGB colour channels / here we don't
% care so we average
im = mean(im,3);

% make the image square either by zero padding or truncate
N = min(size(im));
index = (max(size(im)) - N) / 2;
im = im((1+index):size(im,1)-index,:);

% check the image ;
figure('Name','Roberto'); imagesc(im); title('Original image','Fontsize',14)
xlabel('Pixel number','Fontsize',12); ylabel('Pixel number','Fontsize',12);
colormap('gray')

% compute the fft using the ff2 function + use the fftshift function to shift
% the 0 frequency component (DC) from top left corner to the center
imf=fftshift(fft2(im));

% just as for 1D signal the frequencies are half the length of the signal
figure('Name','2D Fourier transform of Roberto')
freq =-N/2:N/2-1; subplot(2,2,1);
imagesc(freq,freq,log(abs(imf)));
title('FFT','Fontsize',14)
xlabel('Freqencies','Fontsize',12); ylabel('Frequencies','Fontsize',12);

% To obtain a finer sampling of the Fourier transform,
% add zero padding to the image (more pixels = more frequencies)
% when computing its DFT. The zero padding and DFT computation
% can be performed in a single step
imf2=fftshift(fft2(im,256,256)); % zero-pads im to be 256-by-256
freq =-256/2:256/2-1; subplot(2,2,2);
imagesc(freq,freq,log(abs(imf2)));
title('FFT at higher resolution','Fontsize',14)
xlabel('Freqencies','Fontsize',12); ylabel('Frequencies','Fontsize',12);

% power spectrum (ignore the vertical and horizontal streak - its
% an artifact due to the boundaries of the image).
impf=abs(imf2).^2; subplot(2,2,3);
imagesc(freq,freq,log10(impf)); title('Power spectrum','Fontsize',14)
xlabel('Freqencies','Fontsize',12); ylabel('Frequencies','Fontsize',12);

% rotational average to get the profile of the power along the freq axis
% for a given frequency coordinate in x and y we have a power value for
% instance at freq(0,0) impf(128,128) = 1.5293e+12 ; we can express this
% coordinate [x,y] = (128,128) in polar coordinates using cart2pol giving
% theta the angle between the x axis and the vector (0,0)(128,128) and rho
% the length of this vector - once in polar coordinates, reading rho is
% like reading the frequency value in the image - theta can be used to
% investigate which frequencies are in phase

[X Y]=meshgrid(freq,freq); % get coordinates of the power spectrum image
[theta rho]=cart2pol(X,Y); % equivalent in polar coordinates

rho=round(rho);
f=zeros(256/2+1);
for r=0:256/2
    i{r+1}=find(rho==r); % for each freq return the location in the polar
    % array ('find') of the corresponding frequency
    f(r+1)=mean(impf(i{r+1})); % average power values of all the same freq
end

% freqency spectrum
freq2=0:256/2; subplot(2,2,4);
loglog(freq2,f); title('frequency spectrum','Fontsize',14); axis tight
xlabel('Freqencies','Fontsize',12); ylabel('Power','Fontsize',12);

% rough frequency filtering and inverse fourier transform
% imf2 contains the freqencies of the image with freq(0,0) at the middle
figure('Name','Inverse Fourier transforms of Roberto')

bound = round((N/10)/2);
low_freq = [N/2-bound:N/2+bound]; % from middle to image expand 1/10
b = zeros(N,N); b(low_freq,low_freq) = 1;
subplot(3,2,1); imagesc(b);
subplot(3,2,2); imagesc(real(ifft2(fftshift(fft2(im)).*b)));
title('Roberto low freq','Fontsize',14);

middle_freq = [N/2-5*bound:N/2+5*bound];
b2 = zeros(N,N); b2(middle_freq,middle_freq) = 1; b2 = b2 - b;
subplot(3,2,3); imagesc(b2);
subplot(3,2,4); imagesc(real(ifft2(fftshift(fft2(im)).*b2)));
title('Roberto mid freq','Fontsize',14);

b3 = ones(N,N) - b2 - b;
subplot(3,2,5); imagesc(b3);
subplot(3,2,6); imagesc(real(ifft2(fftshift(fft2(im)).*b3)));
title('Roberto high freq','Fontsize',14);
colormap('gray')

%% filtering again using a filter

% say we want to filter up to 10Hz (low-pass)
unit = 1/(N/2);
ffilter = 10*unit;
imf=fftshift(fft2(im));

for i=1:N
    for j=1:N
        r2=(i-round(N/2))^2+(j-round(N/2))^2;
        if r2>round((N*ffilter)^2)
            imf(i,j)=0;
        end
    end
end

Ifilter=real(ifft2(fftshift(imf)));
figure('Name','frequency filtering')
subplot(1,3,1); imagesc(im); title('Original Image','Fontsize',14)
subplot(1,3,2); imagesc(Ifilter); title('10Hz low-pass','Fontsize',14)
subplot(1,3,3); imagesc(im-Ifilter); title([num2str(N/2-10) 'Hz High-pass'],'Fontsize',14);
colormap('gray')


%% filtering using convolution
filter = ones(5,5);
filtered_im = (convn(im,filter))./numel(filter);
padding = (size(filtered_im)-N)/2; N2 = size(filtered_im,1);
filtered_im = filtered_im(padding:(N2-padding-1),padding:(N2-padding-1));
figure;
subplot(2,2,1); imagesc(uint8(filtered_im)); title('Convolution','Fontsize',14)
subplot(2,2,2); imagesc(uint8(im-filtered_im)+127);

% the dedicated function is
h = fspecial('average', [5 5]);
Y = filter2(h,im);
subplot(2,2,3); imagesc(Y); title('Filter','Fontsize',14)
subplot(2,2,4); imagesc(im-Y); colormap('gray')

%% phase scrambling

clear all
im=mean(double(imread('Roberto.jpg')),3);
N = min(size(im));
index = (max(size(im)) - N) / 2;
im = im((1+index):size(im,1)-index,:);
imf=fftshift(fft2(im));
freq =-N/2:N/2-1; 
impf=abs(imf).^2; 
[X Y]=meshgrid(freq,freq); 
[theta rho]=cart2pol(X,Y); 
rho=round(rho);
f=zeros(N/2+1);
for r=0:N/2
    i{r+1}=find(rho==r); 
    f(r+1)=mean(impf(i{r+1})); 
end
figure; 
subplot(2,4,1); imagesc(im); title('original','Fontsize',14);
subplot(2,4,2); imagesc(angle(imf)); title ('Phase','Fontsize',14);
subplot(2,4,3); hist(im(:)); 
mytitle = sprintf('mean %s, std %s \n skewness %s kurtosis %s', mean(im(:)), std(im(:)), skewness(im(:)), kurtosis(im(:)));
title(mytitle)
subplot(2,4,4); freq2=0:N/2; loglog(freq2,f); title('frequency spectrum','Fontsize',14); axis tight


%generate random phase structure
RandomPhase = angle(fft2(rand(N, N)));
ImFourier = fft2(im);
Amp = abs(ImFourier);
Phase = angle(ImFourier);
Phase = Phase + RandomPhase;
ImScrambled = ifft2(Amp.*exp(sqrt(-1)*(Phase)));
ImScrambled = real(ImScrambled);

N = min(size(ImScrambled));
index = (max(size(ImScrambled)) - N) / 2;
im = ImScrambled((1+index):size(ImScrambled,1)-index,:);, 
imf=fftshift(fft2(im));
freq =-N/2:N/2-1; 
impf=abs(imf).^2; 
[X Y]=meshgrid(freq,freq); 
[theta rho]=cart2pol(X,Y); 
rho=round(rho);
f=zeros(N/2+1);
for r=0:N/2
    i{r+1}=find(rho==r); 
    f(r+1)=mean(impf(i{r+1})); 
end
subplot(2,4,5); imagesc(ImScrambled); title('scrambled','Fontsize',14);
subplot(2,4,6); imagesc(angle(imf)); title ('Phase','Fontsize',14);
subplot(2,4,7); hist(ImScrambled(:)); 
mytitle = sprintf('mean %s, std %s \n skewness %s kurtosis %s', mean(im(:)), std(im(:)), skewness(im(:)), kurtosis(im(:)));
title(mytitle)
subplot(2,4,8); freq2=0:N/2; loglog(freq2,f); title('frequency spectrum','Fontsize',14); axis tight
imwrite(ImScrambled,'Roberto_scrambled.jpg','jpg');




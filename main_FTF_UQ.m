%%%%%%%%%% main_FTF_UQ %%%%%%%%%%
% This script performs uncertainty quantification on the estimate of the
% Impulse Reponse Function and Flame Transfer Function (FTF) of a Laminar
% premixed slit flame. First, the impulse response coefficients are
% estimated from time series data. The uncertainty of the estimated
% coefficients is given by the variance matrix. From the impulse response
% coefficients, the FTF coefficients are computed. The variance of the FTF
% coefficients is computed from the variance of the IR coefficients. Using
% the variance matrix of the FTF coefficients, the Rice and Rician Phase
% probability distributions at each sample frequency are computed. Using
% the defined z-score, confidence intervals are then computed. The results
% are plotted at the end.

% Author: Florian Radack (2024), CAPS, ETH Zürich
% last edit: 06.03.2025

clear; close all; clc;

%%
% add data directory
addpath(genpath('../Data/CH4/'))

%% Z-Score
% set z-score for statistical analysis in entire script

z = 2.58;  % z-score
CI = 100*round(normcdf(z),3);  % confidence intercal in percent

%% Gather and Pre-Process Data

% instatiate SYSID class (creat SYSID object)
sys_TI = SYSID('U0.4');

% read AVBP temporal files
sys_TI = read_data(sys_TI, 'umax_Inlet.dat', 'HR_mean.dat'); 

% compute mean velocity and HRR
sys_TI = get_trend(sys_TI, 'constant'); 

% resample data
sys_TI = resample_data(sys_TI, 5e-4);

% normalize and detrend data
sys_TI = normalize_data(sys_TI, 'detrend');  
 
% truncate data
sys_TI = truncate_data(sys_TI, 0.05);

%% OLS Regression

% information criteria to choose model order
min_crit = 'nAIC';

M = length(sys_TI.t);
Ll = 10;  % lowest  model order
Lu = 80;  % highest model order
LL = Ll:Lu;

% preallocate for BIC and nAIC matrices
BIC = zeros(Lu-Ll, 1);
nAIC = zeros(Lu-Ll, 1);

for j = 1:length(LL)
    % construct regression matrices
    sys_TI = create_Phi(sys_TI, LL(j), M);
    sys_TI = create_Y(sys_TI, LL(j), M, 'feedthrough');

    % estimate impulse reponse coefficients
    sys_TI = ti_impulseest(sys_TI);

    % % residuals
    % eps = Phi*IR - Y;

    % Information Criteria
    nAIC(j) = log(sum(sys_TI.eps.^2)/sys_TI.N) + 2*LL(j)/sys_TI.N;
    BIC(j) = sys_TI.N*log(sum(sys_TI.eps.^2)/sys_TI.N) + LL(j)*log(sys_TI.N);

end

% set model order that minimizes information criterium
switch min_crit
    case 'nAIC'
        [nAICmin, nAICmin_idx] = min(nAIC);
        L = LL(nAICmin_idx);
        disp('Setting model order using nAIC.')

    case 'BIC'
        [BICmin, BICmin_idx] = min(BIC);
        L = LL(BICmin_idx);
        disp('Setting model order using BIC.')
end

% overwrite L / set user defined value for L
L = 40; 

% recompute regression matrices from optimal parameters
sys_TI = create_Phi(sys_TI, L, M);
sys_TI = create_Y(sys_TI, L, M, 'feedthrough');

% estimate time-invariant impulse response
sys_TI = ti_impulseest(sys_TI);

%% plot impulse response

figure('Name', 'IRF', 'Units', 'centimeters', 'Position', [0, 0, 9.0, 6.0])
set(gca,'DefaultLineLineWidth',1) 
set(gca,'FontSize',12)
set(gcf, 'Color', 'white')
clf
hold on
scatter(1000*sys_TI.tau, sys_TI.H, 'rs', 'filled')
patch([sys_TI.tau, fliplr(sys_TI.tau)]  , [sys_TI.H-z*sqrt(diag(sys_TI.var_H)); flipud(sys_TI.H+z*sqrt(diag(sys_TI.var_H)))], ...
    [0, 0.4470, 0.7410], 'FaceAlpha',0.2, 'EdgeColor', 'none')

errorbar(1000*sys_TI.tau, sys_TI.H, -z*sqrt(diag(sys_TI.var_H)), +z*sqrt(diag(sys_TI.var_H)), 'r', 'LineStyle', 'none')
plot(1000*sys_TI.tau, sys_TI.H, 'k')
xlabel('tau [ms]')
ylabel('h')
yline(0,'-.k')
xlim([0, 1000*0.0105])
% print('-vector','-dsvg','IRF')

%% Compute Transfer Function from IRF
Nfft = L;
sys_TI =  freqresp(sys_TI, L);

% cutoff frequency of forcing signal
f_c = 600;  

%% Compute Probability Distriubtions of Gain and Phase of Fourier Coefficents
% The distributions follow a Rice distribution (Gain) and Rician Phase
% Distribution (Phase), except at the 0 frequency and the Nyquist
% frequency, where the Gain is a folded-normal distribution and the phase
% is a Dirac distribution.

% create vector for absolute value r and phase angle phi
r = linspace(0, 2, 10000);
phi = linspace(-pi, pi, 10000);

% indices of folded-normal fourier coefficients
idx_fold = [1, Nfft/2+1];  % 0 and Nyquist frequency
idx_fold = idx_fold(mod(idx_fold,1) == 0);  % only retain Nyquist frequency is Nfft is even (k is integer)

% parameters of distribution nu^2 and sig^2
nu2 = abs(sys_TI.F2).^2;
% sig2 = ones(L,1)*sum(diag(var_h));  % if var_h is diagonal
sig2 = 0.5*real(diag(sys_TI.var_F));  % variance of each component of complex distribution is 1/2 of total variance
sig2(idx_fold) = 2*sig2(idx_fold);    % correction for folded-normal distribution


% compute Rice PDF and CDF
pdfRice = SYSUQ.ricePDF(r, abs(sys_TI.F2), sqrt(sig2));
cdfRice = zeros(L, length(r));
for q = 1:Nfft
    cdfRice(q,:) = SYSUQ.riceCDF(r, abs(sys_TI.F2(q)), sqrt(sig2(q)));
end


% compute Folded Normal PDF and CDF
pdfFnormal = SYSUQ.fnormalPDF(r, abs(sys_TI.F2(idx_fold)), sqrt(sig2(idx_fold)));
cdfFnormal = SYSUQ.fnormalCDF(r, abs(sys_TI.F2(idx_fold)), sqrt(sig2(idx_fold)));


% compute Rician Phase PDF and CDF
pdfRicephase = SYSUQ.ricephasePDF(phi, abs(sys_TI.F2), angle(sys_TI.F2), sqrt(sig2));

% compute relative distribution (centered at mode angle)
[~, idx_angshft] = min(abs(phi - angle(sys_TI.F2)), [], 2);
pdfRicephase_rel = pdfRicephase;
for l = 1:Nfft
    pdfRicephase_rel(l, :) = circshift(pdfRicephase(l,:), -(idx_angshft(l)+length(phi)/2));
end

cdfRicephase = SYSUQ.ricephaseCDF(phi, pdfRicephase);
cdfRicephase_rel = SYSUQ.ricephaseCDF(phi, pdfRicephase_rel);


% compute mean and variance of distributions
[meanRice, varRice] = SYSUQ.ricestat(sqrt(nu2), sqrt(sig2));
[meanFnormal, varFnormal] = SYSUQ.foldednormalstat(sqrt(nu2(idx_fold)), sqrt(sig2(idx_fold)));


% combine different PDFs/CDFS into single array
pdfGain = pdfRice;
pdfGain(idx_fold,:) = pdfFnormal;
cdfGain = cdfRice;
cdfGain(idx_fold,:) = cdfFnormal;

pdfPhase = pdfRicephase;
cdfPhase = cdfRicephase;
pdfPhase_rel = pdfRicephase_rel;
cdfPhase_rel = cdfRicephase_rel;

meanGain = meanRice;
meanGain(idx_fold) = meanFnormal;

varGain = varRice;
varGain(idx_fold) = varFnormal;


% clear distribution variables to clear workspace
clear pdfRice cdfRice pdfFnormal cdfFnormal pdfRicephase pdfRicephase_rel ...
    cdfRicephase cdfRicephase_rel meanRice varRice meanFnormal varFnormal

%% Compute Confidence Intervals
% find argument indices of CDF that correspond to the confidence interval
% defined by z-score

idx_r_CI = zeros(Nfft, 2);      % pre-allocate array
idx_phi_CI = ones(Nfft, 2);     % pre-allocate array

% loop over all frequencies to compute indices of each frequency
for l = 1:Nfft
    [~, idx_abs] = min(abs(cdfGain(l,:)  - [(100-CI)/2; CI + (100-CI)/2]/100), [], 2);
    [~, idx_ang] = min(abs(cdfPhase_rel(l,:) - [(100-CI)/2; CI + (100-CI)/2]/100), [], 2);

    idx_r_CI(l,:) = idx_abs;
    idx_phi_CI(l,:) = idx_ang;
end

% reduce array of indices to those of single-sided spectrum
if mod(Nfft, 2) == 0
    idx_r_CI = idx_r_CI(1:Nfft/2+1,:);
    idx_phi_CI = idx_phi_CI(1:Nfft/2+1,:);
else
    idx_r_CI = idx_r_CI(1:(Nfft+1)/2,:);
    idx_phi_CI = idx_phi_CI(1:(Nfft+1)/2,:);
end

%% Comparison to sampled Impulse Response response

Nsample = 100000;  % number of samples drawn from each IR coef distribution
IR_samp = mvnrnd(sys_TI.H, sys_TI.var_H, Nsample)';  % draw samples
H2_samp = fft(IR_samp, Nfft, 1);  % compute TF coefficients
H_samp = H2_samp(1:floor(Nfft/2+1),:);  % take single-sided spectrum

% Compute percentile/confidence intervals from samples
H_samp_mean_abs = mean(abs(H_samp), 2);
H_samp_prc_abs = prctile(abs(H_samp), [(100-CI)/2; CI + (100-CI)/2], 2);
H_samp_prc_ang = prctile(angle(H_samp), [(100-CI)/2; CI + (100-CI)/2], 2);

%% Plot sample histograms and probability distributions

figure('Name', 'Histograms and PDF of FTF Coefficients', 'Units', 'centimeters', 'Position', [0, 0, 19.0, 19.0])
set(gcf, 'color', 'w')

% number of FTF coefficients to plot
Nplot = 11;

for l = 1:Nplot
    subplot(Nplot,2,2*l-1)

    % plot histogram of sampled data
    % h = histogram((abs(H2_samp(l,:)) - mean(abs(H2_samp(l,:))))/sqrt(sig2(l)), 200, 'Normalization', 'pdf', 'EdgeColor','none');
    h = histogram(abs(H2_samp(l,:)), 200, 'Normalization', 'pdf', 'EdgeColor','none');

    hold on
    plot(r, pdfGain(l,:), 'LineWidth',2)

    % center plot at mode
    [~,idx_mode] = max(h.Values);
    % xlim(0.5*(h.BinEdges(idx_mode) + h.BinEdges(idx_mode+1)) + [-0.4,0.4])
    xlim([0 1.6])

    % set background to white
    set(gca, 'YTick', [])

    % set title
    if l==1
        title('pdf(r)')
    end

    % plot properties
    ylabel(strcat('\omega_{', num2str(l), '}'), 'rotation',0)
    box off

end
% legend('', 'pdf[R(\mu,\sigma)]')

for l = 1:Nplot
    subplot(Nplot,2,2*l)

    % plot histogram of sampled data
    h = histogram(angle(H2_samp(l,:)), 200, 'Normalization', 'pdf', 'EdgeColor','none');

    hold on
    if any(l == idx_fold)
    else
        % plot rice distribution
        plot(phi, pdfPhase(l,:), 'LineWidth',2)
    end

    % center plot at mode
    [~,idx_mode] = max(h.Values);
    % xlim(0.5*(h.BinEdges(idx_mode) + h.BinEdges(idx_mode+1)) + [-0.3,0.3])  

    % set background to white
    set(gca, 'YTick', [])

    % set title
    if l==1
        title('pdf(\phi)')
    end

    % plot properties
    xticks(gca, [-pi, -pi/2, 0, pi/2, pi])
    xticklabels(gca, {'-\pi', '-\pi/2', '0', '\pi/2', '\pi'})
    box off

end
%%
figure('Name', 'FTF Error Bars', 'Units', 'centimeters', 'Position', [0, 0, 9.0, 9.0])
set(gca,'DefaultLineLineWidth',1) 
set(gca,'FontSize',12)
set(gcf, 'Color', 'white')
wdth_hbar = 10;

% compute zero-padded FTF
Nfft = 2^(nextpow2(L)+3);
F2 = fft(sys_TI.H, Nfft, 1);  
            
% take single-sided spectrum
if mod(Nfft, 2) == 0
    F = F2(1:Nfft/2+1, :);
else
    F = F2(1:(Nfft+1)/2, :);
end

% frequency vector
if mod(Nfft, 2) == 0
    % N is even
    f = (0:(Nfft/2))/Nfft/sys_TI.dt; 
else
    % N is odd
    f = (0:((Nfft-1)/2))/Nfft/sys_TI.dt;
end

%%% plot gain
subplot(211)
hold on
plot(f, abs(F), 'k', 'LineWidth', 1)

% plot errorbar (99%-CI)
plot([sys_TI.f; sys_TI.f], [r(idx_r_CI(:,1)); r(idx_r_CI(:,2))], 'r')
% horizonal bar
plot([sys_TI.f-wdth_hbar; sys_TI.f+wdth_hbar], [r(idx_r_CI(:,1)); r(idx_r_CI(:,1))], 'r')
plot([sys_TI.f-wdth_hbar; sys_TI.f+wdth_hbar], [r(idx_r_CI(:,2)); r(idx_r_CI(:,2))], 'r')

% plot properties
xlim([0, f_c+wdth_hbar])
xlabel('f [Hz]')
ylabel('|F|')


%%% plot phase
subplot(212)
hold on
plot(f, angle(F), 'k', 'LineWidth', 1)

% wrap error bars 
for i= 2:length(sys_TI.F)
    if angle(sys_TI.F(i)) + phi(idx_phi_CI(i,1)) > -pi && angle(sys_TI.F(i)) + phi(idx_phi_CI(i,2)) < pi
        % errorbar
        plot([sys_TI.f(i), sys_TI.f(i)], [ angle(sys_TI.F(i)) + phi(idx_phi_CI(i,1)), angle(sys_TI.F(i)) + phi(idx_phi_CI(i,2)) ] , 'r')

        % horizonal bar
        plot([sys_TI.f(i)-wdth_hbar, sys_TI.f(i)+wdth_hbar], [angle(sys_TI.F(i)) + phi(idx_phi_CI(i,1)), angle(sys_TI.F(i)) + phi(idx_phi_CI(i,1))], 'r')
        plot([sys_TI.f(i)-wdth_hbar, sys_TI.f(i)+wdth_hbar], [angle(sys_TI.F(i)) + phi(idx_phi_CI(i,2)), angle(sys_TI.F(i)) + phi(idx_phi_CI(i,2))], 'r')

    elseif angle(sys_TI.F(i)) + phi(idx_phi_CI(i,1)) < -pi
        % errorbar
        plot([sys_TI.f(i), sys_TI.f(i)], [-pi, angle(sys_TI.F(i)) + phi(idx_phi_CI(i,2)) ] , 'r')
        plot([sys_TI.f(i), sys_TI.f(i)], [angle(sys_TI.F(i)) + phi(idx_phi_CI(i,1))+2*pi, pi] , 'r')

        % horizontal bar
        plot([sys_TI.f(i)-wdth_hbar, sys_TI.f(i)+wdth_hbar], [angle(sys_TI.F(i)) + phi(idx_phi_CI(i,1))+2*pi, angle(sys_TI.F(i)) + phi(idx_phi_CI(i,1))+2*pi], 'r')
        plot([sys_TI.f(i)-wdth_hbar, sys_TI.f(i)+wdth_hbar], [angle(sys_TI.F(i)) + phi(idx_phi_CI(i,2)), angle(sys_TI.F(i)) + phi(idx_phi_CI(i,2))], 'r')
    else
        % errorbar
        plot([sys_TI.f(i), sys_TI.f(i)], [angle(sys_TI.F(i)) + phi(idx_phi_CI(i,1)), pi ] , 'r')
        plot([sys_TI.f(i), sys_TI.f(i)], [-pi, angle(sys_TI.F(i)) + phi(idx_phi_CI(i,2))-2*pi] , 'r')

        % horizonal bar
        plot([sys_TI.f(i)-wdth_hbar, sys_TI.f(i)+wdth_hbar], [angle(sys_TI.F(i)) + phi(idx_phi_CI(i,2))-2*pi, angle(sys_TI.F(i)) + phi(idx_phi_CI(i,2))-2*pi], 'r')
        plot([sys_TI.f(i)-wdth_hbar, sys_TI.f(i)+wdth_hbar], [angle(sys_TI.F(i)) + phi(idx_phi_CI(i,1)), angle(sys_TI.F(i)) + phi(idx_phi_CI(i,1))], 'r')
    end
end

xlabel('f [Hz]')
ylabel('aF')
yline(-pi, ':'); yline(pi, ':')
xlim([0, f_c + wdth_hbar])
yticks([-pi, 0 , pi]);
yticklabels({'-pi', '0' , 'pi'});
ylim([-pi-0.5, pi+0.5])

% print('-vector','-dsvg','FTF_errorbars')

%% 3D-Plot Marginal Probability Distribution
idx_fc = 2;
cmap = abyss(2);
figure('Units', 'centimeters', 'Position', [0, 0, 9.0, 9.0]);
clf
hold on
plot3(real(H2_samp(idx_fc,1:300).'), imag(H2_samp(idx_fc,1:300).'), zeros(size(H2_samp(idx_fc,1:300))), '.', 'Color', [0.7, 0.7, 0.7])
plot3(real(sys_TI.F2(idx_fc)), imag(sys_TI.F2(idx_fc)), 0.1, 'o', 'Color', 'red', 'Markersize', 5, 'MarkerFaceColor', 'red')
plot3(abs(sys_TI.F2(idx_fc)).*cos(phi), abs(sys_TI.F2(idx_fc)).*sin(phi), pdfPhase(idx_fc,:), 'Color', cmap(1,:), 'LineWidth',1)
plot3(r.*cos(angle(sys_TI.F2(idx_fc))), r.*sin(angle(sys_TI.F2(idx_fc))), pdfGain(idx_fc,:), 'Color', cmap(2,:), 'LineWidth',1)
surf([abs(sys_TI.F2(idx_fc)).*cos(phi); abs(sys_TI.F2(idx_fc)).*cos(phi)].',  ...
     [abs(sys_TI.F2(idx_fc)).*sin(phi); abs(sys_TI.F2(idx_fc)).*sin(phi)].',  ...
     [zeros(size(phi)); pdfPhase(idx_fc,:)].', 'FaceColor', cmap(1,:), 'EdgeColor', 'none', 'FaceAlpha', 0.2)
surf([r.*cos(angle(sys_TI.F2(idx_fc))); r.*cos(angle(sys_TI.F2(idx_fc)))].',  ...
     [r.*sin(angle(sys_TI.F2(idx_fc))); r.*sin(angle(sys_TI.F2(idx_fc)))].',  ...
      [zeros(size(phi)); pdfGain(idx_fc,:)].', 'FaceColor', cmap(2,:), 'EdgeColor', 'none', 'FaceAlpha', 0.2)
ax = gca;
ax.XAxis.FirstCrossoverValue  = 0;
ax.XAxis.SecondCrossoverValue = 0;
ax.YAxis.FirstCrossoverValue  = 0;
ax.YAxis.SecondCrossoverValue = 0;
ax.ZAxis.FirstCrossoverValue  = 0;
ax.ZAxis.SecondCrossoverValue = 0;
xlim([real(sys_TI.F2(idx_fc))-0.3, real(sys_TI.F2(idx_fc))+0.3])
ylim([imag(sys_TI.F2(idx_fc))-0.3, imag(sys_TI.F2(idx_fc))+0.3])

set(gca, 'ZTick', [])
set(gcf, 'Color', 'w')
view(-190, 40)
% export_fig dist.svg -painters

%% Joint and Marginal Probability Distribution, Confidence Intervals
idx_fc = [9,11];
cmap = {brewermap(256, 'blues'), brewermap(256, 'greens')};
ls = {'-.', '-'};

fig2 = figure('Units', 'centimeters', 'Position', [0, 0, 19.0, 6.0]);
clf
set(gca,'DefaultLineLineWidth',1) 
set(gca,'FontSize',12)
set(gcf, 'Color', 'white')

% create tiled layoutout with compacted white-space
t = tiledlayout(2,3);
t.TileSpacing = 'compact';
t.Padding = 'compact';

for i=1:length(idx_fc)
    
    % cartesian coordinates for surface plot of joint-pdf
    x = real(sys_TI.F2(idx_fc(i))) + linspace(-0.2, 0.2, 100);
    y = imag(sys_TI.F2(idx_fc(i))) + linspace(-0.2, 0.2, 100);

    % mesh-grid from coordinates for surface plot
    [xx, yy] = meshgrid(x, y);

    % joint-pdf in cartesian coordinates
    pdfjoint_cart = 1./(2*pi*sig2(idx_fc(i))) .* exp(-1/(2*sig2(idx_fc(i))) .* ...
        ( (xx - real(sys_TI.F2(idx_fc(i)))).^2 + (yy - imag(sys_TI.F2(idx_fc(i)))).^2 ) ) ;
        
    % surface plot tiles
    ax = nexttile(2*i-1, [2,1]);
    hold on

    % plot surface of joint-pdf
    pc = pcolor(xx, yy, pdfjoint_cart);
    pc.FaceColor = 'interp';
    pc.EdgeColor = 'none';
    colormap(ax, cmap{i})

    % scatter plot of  sampled FTF and true FTF
    scatter(real(H2_samp(idx_fc(i),1:1000).'), imag(H2_samp(idx_fc(i),1:1000).'), 2, 'k', 'filled')
    scatter(real(sys_TI.F2(idx_fc(i))), imag(sys_TI.F2(idx_fc(i))), 40, 'red', 'filled', 's')

    % plot levelset for 99% confidence interval
    plot(real(sys_TI.F2(idx_fc(i))) + z*sqrt(sig2(idx_fc(i))) .* cos(phi), imag(sys_TI.F2(idx_fc(i))) + z*sqrt(sig2(idx_fc(i))) .* sin(phi), 'k--')

    % plot 99% confidence of marginal distributions
    r99 = [r(idx_r_CI(idx_fc(i),1)), r(idx_r_CI(idx_fc(i),2))] .* [real(sys_TI.F(idx_fc(i))); imag(sys_TI.F(idx_fc(i)))] / abs(sys_TI.F(idx_fc(i)));
    r99_bar_1 = r99(:,1) + 0.01*[-1 1; -1 1].*[-imag(sys_TI.F(idx_fc(i))); real(sys_TI.F(idx_fc(i)))] / abs(sys_TI.F(idx_fc(i)));
    r99_bar_2 = r99(:,2) + 0.01*[-1 1; -1 1].*[-imag(sys_TI.F(idx_fc(i))); real(sys_TI.F(idx_fc(i)))] / abs(sys_TI.F(idx_fc(i)));

    % exponential form 99% confidence interval of phase
    phi99 = abs(sys_TI.F(idx_fc(i))) .* exp( 1i*(angle(sys_TI.F(idx_fc(i))) ...
        + [phi(idx_phi_CI(idx_fc(i),1)):0.01:phi(idx_phi_CI(idx_fc(i),2))]) );
    phi99_bar_1 = [real(phi99(:,1)); imag(phi99(:,1))] ...
        + 0.01*[-1 1; -1 1].*[real(abs(sys_TI.F(idx_fc(i)))*exp( 1i*(angle(sys_TI.F(idx_fc(i)))...
        +phi(idx_phi_CI(idx_fc(i),1))))); imag(abs(sys_TI.F(idx_fc(i)))*exp( 1i*(angle(sys_TI.F(idx_fc(i)))...
        +phi(idx_phi_CI(idx_fc(i),1)))))] / abs(sys_TI.F(idx_fc(i)));
    phi99_bar_2 = [real(phi99(:,end)); imag(phi99(:,end))] ...
        + 0.01*[-1 1; -1 1].*[real(abs(sys_TI.F2(idx_fc(i)))*exp( 1i*(angle(sys_TI.F(idx_fc(i)))...
        +phi(idx_phi_CI(idx_fc(i),2))))); imag(abs(sys_TI.F(idx_fc(i)))*exp( 1i*(angle(sys_TI.F(idx_fc(i)))...
        +phi(idx_phi_CI(idx_fc(i),2)))))] / abs(sys_TI.F(idx_fc(i)));

    % plot confidence intervals of marginal distributions with error bars
    plot( r99(1,:) , r99(2,:), 'r', 'LineStyle', ls{i} , 'LineWidth', 1)
    plot( real(phi99) , imag(phi99), 'r', 'LineStyle', ls{i} , 'LineWidth', 1)
    plot( r99_bar_1(1,:), r99_bar_1(2,:), 'r', 'LineStyle', '-' , 'LineWidth', 1)
    plot( r99_bar_2(1,:), r99_bar_2(2,:), 'r', 'LineStyle', '-' , 'LineWidth', 1)
    plot( phi99_bar_1(1,:), phi99_bar_1(2,:), 'r', 'LineStyle', '-' , 'LineWidth', 1)
    plot( phi99_bar_2(1,:), phi99_bar_2(2,:), 'r', 'LineStyle', '-' , 'LineWidth', 1)

    % set axis limits to 4 SD
    xlim([real(sys_TI.F2(idx_fc(i))) - 4*sqrt(sig2(idx_fc(i))), real(sys_TI.F2(idx_fc(i))) + 4*sqrt(sig2(idx_fc(i)))])
    ylim([imag(sys_TI.F2(idx_fc(i))) - 4*sqrt(sig2(idx_fc(i))), imag(sys_TI.F2(idx_fc(i))) + 4*sqrt(sig2(idx_fc(i)))])

    % plot and axis settings
    xlabel('Re(z)')
    ylabel('Im(z)')
    zlabel('p(r,p| r,p)')
    xline(0)
    yline(0)
    axis square
    hold off
    

    % plot marginal probability distribution
    nexttile(2)
    [maxmag,idx] = max(pdfGain(idx_fc(i),:));
    hold on
    plot(r, pdfGain(idx_fc(i),:) / maxmag, 'Color', cmap{i}(end/2,:), 'LineWidth',1)
    plot([r(idx_r_CI(idx_fc(i),1)), r(idx_r_CI(idx_fc(i),2))], [0.5+2*i./10, 0.5+2*i./10], 'r', 'LineStyle', ls{i})
    plot([r(idx_r_CI(idx_fc(i),1)), r(idx_r_CI(idx_fc(i),1))], 0.5+2*i./10 + [-0.1, 0.1], 'r')
    plot([r(idx_r_CI(idx_fc(i),2)), r(idx_r_CI(idx_fc(i),2))], 0.5+2*i./10 + [-0.1, 0.1], 'r')
    hold off
    xlabel('r')
    ylabel('p(r|r,p)')
    box off
    xlim([0, 0.3])

    nexttile(5)
    [maxphase,idx] = max(pdfPhase(idx_fc(i),:));
    hold on
    plot(phi, pdfPhase(idx_fc(i),:)/maxphase, 'Color', cmap{i}(end/2,:), 'LineWidth',1)
    plot([max(angle(sys_TI.F(idx_fc(i)))+ phi(idx_phi_CI(idx_fc(i),1)), -pi), min(angle(sys_TI.F(idx_fc(i))) + phi(idx_phi_CI(idx_fc(i),2)), pi)], [0.5+2*i./10, 0.5+2*i./10], 'r', 'LineStyle', ls{i})
    if angle(sys_TI.F(idx_fc(i)))+phi(idx_phi_CI(idx_fc(i),1)) < -pi 
        plot([2*pi + angle(sys_TI.F(idx_fc(i)))+phi(idx_phi_CI(idx_fc(i),1)), pi], [0.5+2*i./10, 0.5+2*i./10], 'r', 'LineStyle', ls{i})
        plot(angle(sys_TI.F(idx_fc(i)))+[phi(idx_phi_CI(idx_fc(i),2)), phi(idx_phi_CI(idx_fc(i),2))], 0.5+2*i./10 + [-0.1, 0.1], 'r')
        plot([2*pi + angle(sys_TI.F(idx_fc(i)))+phi(idx_phi_CI(idx_fc(i),1)), 2*pi + angle(sys_TI.F(idx_fc(i)))+phi(idx_phi_CI(idx_fc(i),1))], 0.5+2*i./10 + [-0.1, 0.1], 'r')
        
    else
        plot(angle(sys_TI.F(idx_fc(i)))+[phi(idx_phi_CI(idx_fc(i),1)), phi(idx_phi_CI(idx_fc(i),1))], 0.5+2*i./10 + [-0.1, 0.1], 'r')
        plot(angle(sys_TI.F(idx_fc(i)))+[phi(idx_phi_CI(idx_fc(i),2)), phi(idx_phi_CI(idx_fc(i),2))], 0.5+2*i./10 + [-0.1, 0.1], 'r')
    end
    hold off
    xlabel('p')
    ylabel('p(p|r,p)')
    box off
    xticks([-pi, 0 , pi]);
    xticklabels({'-pi', '0' , 'pi'});
    xlim([-pi, pi])

end

% print('-vector','-dsvg','marginal_2')

%% FTF in complex plane
fig_FTFcomplex = figure('Units', 'centimeters', 'Position', [0, 0, 7.0, 7.0]);
set(gcf, 'Color', 'white')
hold on
plot(real(F), imag(F), 'k')
scatter(real(sys_TI.F), imag(sys_TI.F), 'r', 's', 'filled')
xline(0)
yline(0)
xlabel('\Re(z)')
ylabel('\Im(z)')
axis equal
xlim([-1.5, 1.5])
yticks([-1, 0 , 1]);

%% Plot of Flame
addpath(genpath('/Users/florianradack/Documents/CAPS/Data/KORNILOV/CH4/'))
fig_FTF = figure('Units', 'centimeters', 'Position', [0, 0, 19.0, 10.0]);
set(gcf,'defaulttextinterpreter','latex');  
set(gcf, 'defaultAxesTickLabelInterpreter','latex');  
set(gcf, 'defaultLegendInterpreter','latex');
set(gca,'DefaultLineLineWidth',1) 
set(gca,'FontSize',12)
set(gcf, 'Color', 'white')

% set colormap by C. Brewer
cmap = brewermap(256, 'blues');
colormap(cmap)

% load coordinates and connectivity
x = double(h5read('U0.2/mesh.mesh.h5', '/Coordinates/x'));
x = x-0.006;
y = double(h5read('U0.2/mesh.mesh.h5', '/Coordinates/y'));
T = double(h5read('U0.2/mesh.mesh.h5', '/Connectivity/tri->node'));
T = reshape(T, 3, [])';  

% load data
HR = double(h5read(sprintf('U%0.1f/last_solution.h5', 0.4), '/Additionals/hr'));

% plot data
trisurf(T,1000*x,1000*y,HR, 'EdgeColor', 'interp')
clim([0 3.5e9])
grid off
shading interp
axis equal
xlim(1000*[0, 0.008])
ylim(1000*[0, 0.0025])
view([-90, 90])

axis off

% set axis on top
set(gca, 'Layer', 'top')
% export_fig flame_0.4 -png
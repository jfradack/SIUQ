classdef SYSUQ
    %SYSUQ class to perform uncertainty quantification on input-output data 
    %   This class can be currently used to:
    % ---------------------------------------------------------------------
    % ---------------------------------------------------------------------
    % last update: 27.05.2024
    % ---------------------------------------------------------------------
    % Authors:
    % Florian Radack   (jradack@ethz.ch)
    % ---------------------------------------------------------------------

    %----------------------------------------------------------------------
    properties
        foldername % name of data folder
        
    end

    %----------------------------------------------------------------------
    methods (Static = true)

        function l = Lhalf(x)
        % Laguerre polynomial L_{1/2}(x)
        % see Moments section of http://en.wikipedia.org/wiki/Rice_distribution
        % l = exp(x/2) * ( (1-x) * besseli(0, -x/2) - x*besseli(1, -x/2) );

        % first compute the log of l to ensure computability of each term
        logl = x/2 + abs(real(-x/2)) + log( (1-x) .* besseli(0, -x/2, 1) - x .* besseli(1, -x/2, 1) );

        % then take the exponential
        l = exp(logl);
        end
        

        function [mu, vr] = ricestat(v, s)
        %RICESTAT Mean and variance of Rice/Rician probability distribution.
        %   [mu vr] = ricestat(v, s) returns the mean and variance of the Rice 
        %   distribution with parameters v and s.

            L = SYSUQ.Lhalf(-0.5 * v.^2 ./ s.^2);
            mu = s .* sqrt(pi/2) .* L;
            vr = 2*s.^2 + v.^2 - (pi * s.^2 / 2) .* L.^2;
        end

        function [mu, vr] = foldednormalstat(v, s)
        %RICESTAT Mean and variance of folded normal probability distribution.
        %   [mu vr] = foldednormalstat(v, s) returns the mean and variance of the 
        %   folded normal distribution with parameters v and s.
        %
        %
        %   Reference: https://en.wikipedia.org/wiki/Folded_normal_distribution

            mu = s .* sqrt(2/pi) .* exp(-0.5*v.^2./s.^2) + v.*erf(v./sqrt(2*s.^2));
            vr = v.^2 + s.^2 - mu.^2;
        end


        function f = ricePDF(x, v, s)
        %RICEPDF Rice/Rician probability density function (pdf).
        %   y = ricepdf(r, v, s) returns the pdf of the Rice (Rician) 
        %   distribution with parameters v and s, evaluated at the values in x.
        
            % square of mean v and standand deviation s
            nu2 = v.^2;
            sig2 = s.^2;
            
            % pdf
            f = x./sig2.*exp(-(x.^2+nu2)/2./sig2 + abs(x.*sqrt(nu2)./sig2)) .* besseli(0,x.*sqrt(nu2)./sig2, 1);
        end


        function f = riceCDF(x, v, s)
        %RICECDF Rice/Rician cumulative distribution function (cdf).
        %   y = ricecdf(x, v, s) returns the cdf of the Rice (Rician) 
        %   distribution with parameters v and s, evaluated at the values in x.
        
            % square of mean v and standand deviation s
            nu2 = v.^2;
            sig2 = s.^2;
            
            % cdf with marcum-q function
            f = 1 - marcumq(sqrt(nu2)./sqrt(sig2), x./sqrt(sig2));
        end


        function f = ricephasePDF(x, r, phi, s)
        %RICEPDF Rician phase probability density function (pdf).
        %   y = ricephasepdf(x, r, phi, s) returns the pdf of the Rician phase 
        %   distribution with parameters r, s and phi, evaluated at the 
        %   values in x.
            
            % pdf
            f = 1/(2*pi) * exp(-r.^2 ./ (2*s.^2)) + ...
                1./(2*s*sqrt(2*pi)) .* (r.*cos(x-phi)) .* ...
                (1+erf(r.*cos(x-phi)./(sqrt(2)*s))) .* ...
                exp(-(r.*sin(x-phi)).^2 ./ (2*s.^2));
        end


        function f = ricephaseCDF(x, f)
        %RICEPDF Rician phase cumulative distribution function (cdf).
        %   y = ricephasecdf(phi, f) returns the cdf of the Rician phase 
        %   distribution with pdf , evaluated at the 
        %   values in x. It cannot be expressed in terms of known special
        %   functions and must be numerically integrated.
            
            % cdf
            f = cumtrapz(x, f', 1)';        
        end


        function f = fnormalPDF(x, v, s)
        %FNORMALPDF Folded normal probability density function (pdf).
        %   y = fnormalPDF(x, v, s) returns the pdf of the folded normal
        %   distribution with parameters v and s, evaluated at the values in x.
        %   Reference: https://en.wikipedia.org/wiki/Folded_normal_distribution
        
            % square of mean v and standand deviation s
            nu2 = v.^2;
            sig2 = s.^2;
            
            % pdf
            f = 1./sqrt(2*pi*sig2) .* exp(- (x-v).^2 ./ (2.*sig2) ) + ...
                1./sqrt(2*pi*sig2) .* exp(- (x+v).^2 ./ (2.*sig2) );
        end


        function f = fnormalCDF(x, v, s)
        %FNORMALCDF Folded normal cumulative distribution function (cdf).
        %   y = fnormalCDF(x, v, s) returns the cdf of the folded normal 
        % distribution with parameters v and s, evaluated at the values in x.
        
            % square of standand deviation s
            sig2 = s.^2;
            
            % cdf
            f = 0.5 * (erf( (x+abs(v)) ./ sqrt(2*sig2) )  + erf( (x-abs(v)) ./ sqrt(2*sig2) ));
        end
        

        function [rl, ru, phil, phiu] = CI_F(Npol, Nfft, F2, var_F, CI)
        %CI_F Confidence intervals of F2
        %   [rl, ru, phil, phiu] = CI_F(Npol, Nfft, F2, var_F, CI) computes
        %   the lower (l) and upper (u) bounds of the confidence interval
        %   of the gain (r) and phase (phi) for the confindence interval
        %   specified by CI. It requires the number of evaluation points in
        %   polar coodinates Npol, the number of points of the DFT Nfft,
        %   the double-sided spectrum F2, the covariance-matrix of the
        %   double sided spectrum var_F, and the confidence level CI. 

            % magnitude r and phase angle phi for comp of CI
            r = linspace(0, 2, Npol);
            phi = linspace(-pi, pi, Npol);
            
            % indices of folded-normal fourier coefficients
            idx_fold = [1, Nfft/2+1]; % 0 and Nyquist frequency
            idx_fold = idx_fold(mod(idx_fold,1) == 0);  % only retain nyquist frequency is Nfft is even (k is integer)
            
            % variance
            sig2 = 0.5*real(diag(var_F));  % variance of each component of complex distribution is 1/2 of total variance
            sig2(idx_fold) = 2*sig2(idx_fold);

            % compute Rice PDF and CDF
            pdfRice = SYSUQ.ricePDF(r, abs(F2), sqrt(sig2));
            cdfRice = zeros(Nfft, length(r));
            for q = 1:Nfft
                cdfRice(q,:) = SYSUQ.riceCDF(r, abs(F2(q)), sqrt(sig2(q)));
            end

            % compute Folded Normal PDF and CDF
            pdfFnormal = SYSUQ.fnormalPDF(r, abs(F2(idx_fold)), sqrt(sig2(idx_fold)));
            cdfFnormal = SYSUQ.fnormalCDF(r, abs(F2(idx_fold)), sqrt(sig2(idx_fold)));

            % compute Rician Phase PDF and CDF
            pdfRicephase = SYSUQ.ricephasePDF(phi, abs(F2), angle(F2), sqrt(sig2));
            % compute relative distribution (centered at mode angle)
            [~, idx_angshft] = min(abs(phi - angle(F2)), [], 2);
            pdfRicephase_rel = pdfRicephase;
            for l = 1:Nfft
                pdfRicephase_rel(l, :) = circshift(pdfRicephase(l,:), -(idx_angshft(l)+length(phi)/2));
            end
            cdfRicephase = SYSUQ.ricephaseCDF(phi, pdfRicephase);
            cdfRicephase_rel = SYSUQ.ricephaseCDF(phi, pdfRicephase_rel);

            % find argument indices for confidence intervals
            idx_r_CI = zeros(Nfft, 2);      % pre-allocate array
            idx_phi_CI = ones(Nfft, 2);     % pre-allocate array
            
            for l = 1:Nfft
                if any(l == idx_fold)
                    [~, idx_abs] = min(abs(cdfFnormal(idx_fold == l, :) - [(100-CI)/2; CI + (100-CI)/2]/100), [], 2);
                    idx_r_CI(l,:) = idx_abs;
                else
                    [~, idx_abs] = min(abs(cdfRice(l,:) - [(100-CI)/2; CI + (100-CI)/2]/100), [], 2);
                    [~, idx_ang] = min(abs(cdfRicephase_rel(l,:) - [(100-CI)/2; CI + (100-CI)/2]/100), [], 2);
            
                    idx_r_CI(l,:) = idx_abs;
                    idx_phi_CI(l,:) = idx_ang;
                end
            end
            
            % indices for single sided spectrum
            idx_r_CI = idx_r_CI(1:floor(Nfft/2+1),:);
            idx_phi_CI = idx_phi_CI(1:floor(Nfft/2+1),:);

            % lower and upper bounds for CI
            rl = r(idx_r_CI(:,1)); 
            ru = r(idx_r_CI(:,2));
            phil = phi(idx_phi_CI(:,1));
            phiu = phi(idx_phi_CI(:,1));
        end
    end

end



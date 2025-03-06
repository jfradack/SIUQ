classdef SYSID
    %SYSID class to perform system identification on input-output data 
    %   This class can be currently used to:
    %       - estimate the stationary impulse response function (IRF)
    %       - estimate the stationary transfer function (TF)
    %       - estimate the time-varying impulse response function (TV-IRF)
    %       - estimate the time-varying transfer function (TV-TF)
    % ---------------------------------------------------------------------
    %   input    ____________   output
    %           |            |
    %   u -->   |    SYS     |   --> y
    %           |____________| 
    %
    % ---------------------------------------------------------------------
    % last update: 23.04.2024
    % ---------------------------------------------------------------------
    % Authors:
    % Florian Radack   (jradack@ethz.ch)
    % ---------------------------------------------------------------------

    %----------------------------------------------------------------------
    properties
        foldername % name of data folder
        u % input data
        y % output data
        t % sampling time
        dt % time step size
        ucoef % polynomial coefficients of trend of u
        ycoef % polynomial coefficients of trend of y
        L % number of impulse response coefficients
        M % number of time steps in data
        N % number of time steps in regressor matrices
        Phi % input regressor matrix
        Y % output regressor vector
        tau % time delay vector
        C % series expansion coefficient matrix
        H % impulse response
        F2 % frequency response (double sided)
        F % frequency response (single sided)
        f % frequency vector
        B % basis function matrix
        I % number of basis functions
        eps % output residual
        var_H  % covariance matrix of IR coefficients
        var_F  % covariance matrix of fourier coefficients
        rl
        ru
        phil
        phiu
        Z
        
    end

    %----------------------------------------------------------------------
    methods
        function obj = SYSID(foldername)
            %SYSID Construct an instance of this class
            obj.foldername = foldername;
        end


        function obj = read_data(obj, filename_u, filename_y)
            % read two-column .csv file containing the sampling time in 
            % the first column and the input or output data in the second
            % column
            
            % read data from file
            data_u = readmatrix(fullfile(obj.foldername, filename_u), 'CommentStyle','#');
            data_y = readmatrix(fullfile(obj.foldername, filename_y), 'CommentStyle','#');

            % extract time from data
            if data_u(:,1) == data_y(:,1) 
                obj.t = data_u(:,1);
            else
                error('Input and output data sampling times do not match.')
            end

            % extract input and output data
            obj.u = data_u(:,2);
            obj.y = data_y(:,2);
        end


        function obj = get_trend(obj, trend, umean, ymean)
        %GET_TREND computes the polynomial coefficients of the trend of the
        %   input and output data. If no second argument is given, a constant
        %   trend is assumed and the coefficients are computed from the mean
        %   of the data. If the data is time-varying, a linear trend can be
        %   specified and the coefficients are computed from the first and
        %   last values of the data.

            obj.ucoef(1) = 0;
            obj.ycoef(1) = 0;
            
            if nargin==1 || strcmpi(trend,'constant')
                % mean value is computed
                obj.ucoef(2) = mean(obj.u);
                obj.ycoef(2) = mean(obj.y);

            elseif nargin==4 & strcmpi(trend,'constant')
                % mean value is specified
                obj.ucoef(2) = umean;
                obj.ycoef(2) = ymean;

            elseif strcmpi(trend,'linear')
                % compute polynomial coefficients to detrend
                obj.ucoef(1) = (obj.u(end)-obj.u(1)) / (obj.t(end)-obj.t(1));
                obj.ucoef(2) = obj.u(1);
            
                obj.ycoef(1) = (obj.y(end)-obj.y(1)) / (obj.t(end)-obj.t(1));
                obj.ycoef(2) = obj.y(1);
            end            
        end


        function obj = resample_data(obj, dt)
        %RESAMPLE_DATA first interpolates the non-uniformly sampled 
        %   data to uniform grid with time step size dt equal to the 
        %   averge time step size. Then, the data is resampled using 
        %   the resample function. Because the resampling function 
        %   assumes that the signal is zero outside the borders of 
        %   the signal, the data is detrended before resamling and 
        %   the trend added back after. The trend is computed by
        %   linearly interpolating between the first and last entry
        %   of the data vectors.

            % step size
            obj.dt = dt;

            % determine time step size for interpolation
            dt_order = abs(round(log10(mean(diff(obj.t)))));    % order of magnitude of average time step size
            dt_mean = round(mean(diff(obj.t)), dt_order);       % average time step size
            t_interp = dt_mean:dt_mean:obj.t(end);              % time vector for interpolation

            % interpolation to constant time step size
            obj.u = interp1(obj.t, obj.u, t_interp);     % interpolated inpput
            obj.y = interp1(obj.t, obj.y, t_interp);     % interpolated output
            
            % resample to new sampling rate and remove trend to avoid edge effects
            obj.u = resample(obj.u - polyval(obj.ucoef,t_interp), 1/obj.dt, 1/dt_mean);
            obj.y = resample(obj.y - polyval(obj.ycoef,t_interp), 1/obj.dt, 1/dt_mean);
            
            % add transient mean back to data
            obj.t = dt_mean:obj.dt:t_interp(end);
            obj.u = obj.u + polyval(obj.ucoef,obj.t);
            obj.y = obj.y + polyval(obj.ycoef,obj.t);
        end


        function obj = normalize_data(obj, detrend)   
        %NORMALIZE_DATA  normalizes data by its trend
            
            % compute trends
            umean = polyval(obj.ucoef,obj.t);
            ymean = polyval(obj.ycoef,obj.t);
            
            % normalize and detrend
            if nargin == 2 && strcmpi(detrend, 'detrend')
              obj.u = (obj.u - umean) ./  umean;
              obj.y = (obj.y - ymean) ./  ymean;
            
            % give error of argument other than detrend is given
            elseif nargin == 2 && ~strcmpi(detrend, 'detrend')
                error("Unrecognized argument '%s'.", detrend)
            
            % normalize without detrending
            else
                obj.u = obj.u  ./  umean;
                obj.y = obj.y  ./  ymean;
            end
        end

        
        function obj = truncate_data(obj, T, offset)
        % TRUNCATE_DATA returns the truncated input and output data
            
            % set offset to zero if not specified
            if nargin == 2
                offset = 0;
            end
            
            % truncate to last T seconds of data
            obj.u = obj.u(end-T/obj.dt+1 - offset/obj.dt:end - offset/obj.dt);
            obj.y = obj.y(end-T/obj.dt+1 - offset/obj.dt:end - offset/obj.dt);
            obj.t = obj.t(end-T/obj.dt+1 - offset/obj.dt:end - offset/obj.dt);
        end


        function obj = remove_window(obj, R)
        % REMOVE_WINDOW returns the input and output data truncated 
        %   to remove the effect from the Tukey window with cosine fraction 
        %   R applied to the input signal.

            % truncate signal
            obj.u = obj.u(R/2*obj.N+1:end-R/2*obj.N);
            obj.y = obj.y(R/2*obj.N+1:end-R/2*obj.N);
            obj.t = obj.t(R/2*obj.N+1:end-R/2*obj.N);
        end


        function obj = create_Phi(obj,L,M)
        %CREATE_PHI creates the Phi regressor matrix
          
            obj.L = L;      % number of impulse response coefficients
            obj.M = M;      % number of data time steps
            obj.N = M-L;    % number of time steps in regressors

            % initialize empty array
            obj.Phi = zeros(L,M-L);
            
            % construct matrix
            for i=1:L
                for j=1:M-L
                        obj.Phi(i,j) = obj.u(L+j-i); 
                end
            end
            
            % transpose array
            obj.Phi = transpose(obj.Phi);
        end


        function obj = create_Y(obj, L, M, feedthrough)
        %CREATE_Y creaes the Y regressor vector. If 'feedthrough' is
        % given as a second argument, direct feedthrough from input to
        % output is assumed. 

            obj.L = L;      % number of impulse response coefficients
            obj.M = M;      % number of data time steps
            obj.N = M-L;    % number of time steps in regressors

            % initialize empty array
            obj.Y = zeros(M-L,1);
        
            % loop over all elements
            for i=1:M-L
                if nargin == 4 && strcmpi(feedthrough, 'feedthrough') 
                    obj.Y(i) = obj.y(L+i-1);  
                    obj.tau = (0:L-1)*obj.dt;  % time delay vector
                else
                    obj.Y(i) = obj.y(L+i);
                    obj.tau = (1:L)*obj.dt;  % time delay vector
                end
            end
        end


        function obj =  tv_impulseest(obj)
        % TV_IMPULSEEST  Nonparametric estimation of time-varying impulse 
        % response (TVIR) from data. The function estimates H from 
        % the input-regressor matrix Phi and the output regressor vector Y 
        % by expanding it over a basis B.
            
            % projection of regressor matrix onto basis beta
            obj.Z = [];
            for i = 1:obj.I
                Zi = obj.Phi.*obj.B(i,:)';
                obj.Z = [obj.Z, Zi];
            end
            
            % compute least-squares estimate of basis coefficients c
            Zeta = obj.Z\obj.Y;
            obj.C = reshape(Zeta, obj.L, obj.I);
            
            % reconstruct kernel
            obj.H = obj.C*obj.B;     

            % estimate residual
            obj.eps = diag(obj.Phi*obj.H) - obj.Y;
        end

        
        function obj =  ti_impulseest(obj)
        % tv_impulseest  Nonparametric estimation of time-invariant impulse 
        %   response IR from data. The function estimates the impulse
        %   response h from the input-regressor matrix Phi and the output 
        %   regressor vector Y. The variance of the impulse response 
        %   coefficients var_h and the residuals eps are also estimated.
            
            % compute least-squares estimate of IR coefficients
            obj.H = obj.Phi \ obj.Y;
            
            % estimate residual
            obj.eps = obj.Phi*obj.H - obj.Y;
            
            % chi-squared estimator of noise variance
            sig = sum(obj.eps.^2)./(obj.N-obj.L);
            
            % variance of H
            obj.var_H = inv(obj.Phi'*obj.Phi) .* sig;            
        end


        function obj = freqresp(obj, N)
        
            % compute fast fourier transform
            obj.F2 = fft(obj.H, N, 1);  
            
            % take single-sided spectrum
            if mod(N, 2) == 0
                obj.F = obj.F2(1:N/2+1, :);
            else
                obj.F = obj.F2(1:(N+1)/2, :);
            end


            % frequency vector
            if mod(N, 2) == 0
                % N is even
                obj.f = (0:(N/2))/N/obj.dt; 
            else
                % N is odd
                obj.f = (0:((N-1)/2))/N/obj.dt;
            end
             
            
            % variance of F
            if size(obj.H, 2) == 1
                W = exp(-2*pi*1i/obj.L * (0:N-1)'.*(0:obj.L-1));
                obj.var_F = W*obj.var_H*W';
            end
        end
        

        function obj = create_basis(obj, I, type)
        % CREATE_BASIS  constructs a matrix of orthonormal basis functions.
        % of order I and discretitized by N points. Each column of B 
        % corresponds to one basis function. 

            % number of basis functions
            obj.I = I;

            % preallocate empty basis function matrix B
            obj.B = zeros(I, obj.N);
            
            switch type
                case 'legendre'  % Legendre polynomial basis
                    for i = 1:obj.I
                        obj.B(i,:) = polyval(LegendrePoly(i-1),linspace(-1,1,obj.N));
                    end
            
                case 'fourier'  % Fourier basis
                    obj.B = exp(-2*pi*1i/obj.N * (0:obj.I-1)'.*(0:obj.N-1));  % non-symmetric DFT basis
                otherwise
                    error("Undefined basis function type. Available options are: 'legendre', 'fourier'.")       
            end
        end
        
    
    end
end
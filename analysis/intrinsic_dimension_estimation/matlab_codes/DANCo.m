function [d,kl] = DANCo(data,varargin)
% DANCO  Estmating the intrinsic dimensionality of a dataset
%
% function [d,kl] = DANCo(data, ParamName,ParamValue, ...)
%
%  This funciton estimates the intrinsic dimensionality of a given dataset
% by means of the DANCo algorithm described in:
%
%  "DANCo: Dimensionality from Angle and Norm Concentration"
%  C. Ceruti, S. Bassis, A. Rozza, G. Lombardi, E. Casiraghi, P. Campadelli
%  arXiv:1206.3881v1, http://arxiv.org/abs/1206.3881v1
%
% with the complexity improvement given by separating the training step
% from the evaluation one (requires fitting models in the file DANCo_fits).
%
%  Parameters
%  ----------
% IN:
%  data     = The data matrix (DxN) with a sample per column.
% Named parameters:
%  'k'
%       The number of neighbours to be used, must be >= 1. (def=10)
%  'minDim'
%       The minimal number of dimensions. (def=1)
%  'maxDim'
%       The maximal number of dimensions. (def=size(data,1))
%  'sigma'
%        Standard deviation of the gaussian smoothing filter executed on
%       the estimated divergences. (def=[]=no smoothing)
%  'fractal'
%       Is the fractal dimension required? (def=false)
%  'mlMethod'
%       Algorithm used to pre-estimate d:
%               MLE         = Levina's algorithm.
%               MiND_MLI    = ML on the first neighbour.
%               MiND_MLK    = ML on the first neighbour and optimization.
%             (def='MiND_MLK')
%  'minCorrectionDim'
%        Minimal dimensionality estimated with the selected mlMethod such
%       that DANCO is adopted for correction, for lower dimensionalities
%       the mlMethod result is produced. (def=5)
%  'inds'
%        Precomputed knn indices as returned by the KNN function for k 
%       set to k+2 with respect to the given k parameter. (also dists must
%       be present)
%  'dists'
%        Precomputed knn distances as returned by the KNN function for k 
%       set to k+2 with respect to the given k parameter. (also inds must
%       be present)
%  'normalized'
%        Are the inds/dists already normalized? Normalized here means that 
%       all the k+2 distances are normalized wrt the last one and the first
%       (zero) and the last (one) are removed. (def=true)
% OUT:
%  d        = Estimated intrinsic dimensionality of the dataset.
%  kl       = The Kullback Leibler divergences curve for d=1..D.
%
%  Examples
%  --------
% X = randsphere(30,1000,1);
% V = linSubspSpanOrthonormalize(randn(120,30));
% pts = V*X;
% [inds,dists] = KNN(pts,12,true);
% [d,kl] = DANCo(pts,'inds',inds,'dists',dists,'minDim',20,'maxDim',40);

    % Checking parameters:
    if nargin<1; error('A dataset is required'); end
    
    % Infos:
    [D,N] = size(data);

    % Default parameters:
    params = struct('k',{10}, ...
                    'minDim',{1}, 'maxDim',{D}, ...
                    'fractal',{false}, ...
                    'sigma',{[]},...
                    'mlMethod',{'MiND_MLK'}, ...
                    'normalized',{true}, ...
                    'minCorrectionDim',{5});
                
    % Parsing named parameters:
    if nargin > 1
        try
            params = parseParamsNamed(varargin, false, params);
        catch e
            error('Pairs ''name,value'' are required as arguments after the dataset.');
        end
    end
    
    % Estimating the parameters for the real data:
    [isLowDim,dHat,mu,tau] = DANCo_statistics(data,params.k,params,nargout > 1);

    % Correction based on the KL-divergence:
    if not(isLowDim) || nargout > 1
        % Initializing the parameters:
        params2 = struct( ...
            'mlMethod',{params.mlMethod}, ...
            'minCorrectionDim',{params.minCorrectionDim});
        numDims = params.maxDim - params.minDim + 1;

        % Computing the reference parameters:
        dHat_ref = zeros(1,numDims);
        mu_ref = zeros(1,numDims);
        tau_ref = zeros(1,numDims);
        for i = 1:numDims
            % Generating the dataset:
            data2 = randsphere(params.minDim + i - 1, N);
            
            % Estimating the parameters:
            [~,dHat_ref(i),mu_ref(i),tau_ref(i)] = ...
                DANCo_statistics(data2,params.k,params2,true);
        end

        % Estimating the KL-divergence for all the cases d=1:D:
        kl = DANCo_estimateKL(params.k,dHat,mu,tau,dHat_ref,mu_ref,tau_ref);
    end
    
    % Smoothing if required:
    if (not(isLowDim) || nargout > 1) && not(isempty(params.sigma))
        % Preparing the impulse response:
        rad = 4*ceil(params.sigma/2);
        h = gauss(-rad:rad,params.sigma,0,false);
        h = h/sum(h);

        % Filtering:
        kl = conv([flipdim(kl,2),kl,kl],h,'same');
        kl = kl(numDims+1:end-numDims);
    end
    
    % The dimensionality estimation:
    if isLowDim
        % Using the MLE estimation:
        if params.fractal; d = dHat;
        else d = round(dHat); end
    else
        % Selecting the minimum kl position:
        [~,d] = min(kl);
        
        % Refining for the fractal dimension:
        if params.fractal
            % Fitting with a cubic smoothing spline:
            fn = csapi(1:numDims,kl);
            
            % Locating the minima:
            opts = optimset('Display','off','TolX',1e-3);
            d = fminsearch(@(x)(fnval(fn,x)),d,opts);
        end
    end
    
    % Correcting the dimension considering the minimal one:
    d = d + params.minDim - 1;
end

function [d,kl] = DANCoFit(data,varargin)
% DANCOFIT  Estmating the intrinsic dimensionality of a dataset
%
% function [d,kl] = DANCoFit(data, ParamName,ParamValue, ...)
%
%  This funciton estimates the intrinsic dimensionality of a given dataset
% by means of the fast DANCo algorithm described in:
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
%  'modelfile'
%        The name of the model file to be used. (def='DANCo_fits')
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
% [d,kl] = DANCoFit(pts,'inds',inds,'dists',dists);

    % Checking parameters:
    if nargin<1; error('A dataset is required'); end
    
    % Infos:
    [D,N] = size(data);
    d = 1:D;

    % Default parameters:
    params = struct('fractal',{false}, ...
                    'mlMethod',{'MiND_MLK'}, ...
                    'normalized',{true}, ...
                    'minCorrectionDim',{5}, ...
                    'modelfile',{'DANCo_fits'});
                
    % Parsing named parameters:
    if nargin > 1
        try
            params = parseParamsNamed(varargin, false, params);
        catch e
            error('Pairs ''name,value'' are required as arguments after the dataset.');
        end
    end
    
    % Loading the model:
    load(params.modelfile);

    % Estimating the parameters:
    [isLowDim,dHat,mu,tau] = DANCo_statistics(data,k,params,nargout > 1);

    % Correction based on the KL-divergence:
    if not(isLowDim) || nargout > 1
        % Computing the reference parameters:
        dHat_ref = feval(fitDhat,d,N);
        mu_ref = feval(fitMu,d,N);
        tau_ref = feval(fitTau,d,N);

        % Estimating the KL-divergence for all the cases d=1:D:
        kl = DANCo_estimateKL(k,dHat,mu,tau,dHat_ref,mu_ref,tau_ref);
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
            fn = csapi(1:D,kl);
            
            % Locating the minima:
            opts = optimset('Display','off','TolX',1e-3);
            d = fminsearch(@(x)(fnval(fn,x)),d,opts);
        end
    end
end

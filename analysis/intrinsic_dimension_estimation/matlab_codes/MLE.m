function [d,ds] = MLE(data,varargin)
%MLE  Intrinsic dimensionality using the Levina's MLE technique
%
% function [d,ds] = MLE(data, ParamName,ParamValue, ...)
%
%  This funciton estimates the intrinsic dimensionality of a dataset using 
% the Levina's Maximum Likelihood Estimator (MLE).
%
%  Parameters
%  ----------
% IN:
%  data     = Matrix DxN containing the dataset as columnwise points.
% Named parameters:
%  'k'
%       The number of neighbours to be used, must be >= 1. (def=10)
%  'dists'
%        Precomputed knn distanced as returned by the KNN function for k 
%       set to k+2 with respect to the given k parameter (k is set to 
%       "size(dists,2)-2" if knn is present).
%  'normalized'
%        Are the dists already normalized? Normalized here means that all
%       the k+2 distances are normalized wrt the last one and the first
%       (zero) and the last (one) are removed. (def=true)
% OUT:
%  d    = Estimated intrinsic dimensionality of the dataset.
%  ds   = Local estimations.
%
%  Examples
%  --------
% X = randsphere(30,1000,1);
% V = linSubspSpanOrthonormalize(randn(120,30));
% pts = V*X;
% [inds,dists] = KNN(pts,12,true);
% d = MLE(pts,'dists',dists);

    % Checking parameters:
    if nargin<1; error('A dataset is required'); end
    
    % Infos:
    N = size(data,2);
    
    % Default parameters:
    params = struct('k',{10},'normalized',{true});
    
    % Parsing named parameters:
    if nargin > 1
        try
            params = parseParamsNamed(varargin, false, params);
        catch e
            error('Pairs ''name,value'' are required as arguments after the dataset.');
        end
    end

    % Is the knn already there?
    mustDoKnn = true;
    if isfield(params,'dists')
        % Extracting and checking:
        dists = params.dists;
        if size(dists,1) == N && size(dists,2) > 2
            % KNN already present:
            params.k = size(dists,2)-2;
            mustDoKnn = false;
        end
    end
    
    % Finding the KNNs:
    if mustDoKnn
        if not(isfield(params,'k'))
            error('One parameter of ''dists'' or ''k'' is required.');
        end
        [~,dists] = KNN(data,params.k,true);
    end
    
    % Normalizing the distances:
    if not(mustDoKnn || params.normalized)
        % Cropping the dists:
        dists = dists(:,2:end);
        
        % Normalizing and removing the last element:
        dists = dists(:,1:end-1)./repmat(dists(:,end),[1,size(dists,2)-1]);
    end

    
    % Local estimation:
    ds = -1./mean(log(dists),2);
    
    % Harmonic mean:
    d = 1./mean(1./ds);
end

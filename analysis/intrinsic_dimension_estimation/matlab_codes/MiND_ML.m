function d = MiND_ML(data,varargin)
%MiND_ML  Intrinsic dimensionality using the MiND_ML techniques
%
% function d = MiND_ML(data, ParamName,ParamValue, ...)
%
%  This funciton estimates the intrinsic dimensionality of a dataset using 
% the Minimum Neighbor Distance estimators described in:
%
% "Minimum Neighbor Distance Estimators of Intrinsic Dimension",
%  A.Rozza, G.Lombardi, C.Ceruti, E.Casiraghi, P.Campadelli,
%  Published to the European Conference of Machine Learning (ECML 2011).
%
%  Parameters
%  ----------
% IN:
%  data     = Matrix DxN containing the dataset as columnwise points.
% Named parameters:
%  'k'
%       The number of neighbours to be used, must be >= 1. (def=10)
%  'optimize'
%        Must an optimization step be executed to estimate a fractal id,
%       that is MiND_MLK instead of MiND_MLI? (def=false)
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
%
%  Examples
%  --------
% X = randsphere(30,1000,1);
% V = linSubspSpanOrthonormalize(randn(120,30));
% pts = V*X;
% [inds,dists] = KNN(pts,12,true);
% d = MiND_ML(pts,'dists',dists);

    % Checking parameters:
    if nargin<1; error('A dataset is required'); end
    
    % Infos:
    [D,N] = size(data);
    
    % Default parameters:
    params = struct('k',{10},'optimize',{false},'normalized',{true});
    
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

    
    % Infos:
    [n,k] = size(dists);
    
    % The first normalized distance:
    r = dists(:,1);
    
    % The log likelihood derivative squared:
    fun = @(d)((n/d)+sum(log(r)-((k-1).*(log(r).*r.^d))./(1-r.^d)))^2;
    
    % Evaluating it:
    vals = zeros(1,D);
    for d=1:D; vals(d)=fun(d); end
    
    % MiND_MLI result:
    [~,d] = min(vals);
    
    % The optimization:
    if params.optimize
        opts = optimset('Display','off','TolX',1e-3);
        d = fminsearch(fun,d,opts);
    end
end

function [d,kl] = MiND_KL(data,varargin)
%MiND_KL  Intrinsic dimensionality using the MiND_KL technique
%
% function [d,kl] = MiND_KL(data, ParamName,ParamValue, ...)
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
%  'minDim'
%       The minimal number of dimensions. (def=1)
%  'maxDim'
%       The maximal number of dimensions. (def=size(data,1))
%  'sigma'
%        Standard deviation of the gaussian smoothing filter executed on
%       the estimated divergences. (def=[]=no smoothing)
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
%  kl   = The Kullback Leibler divergences curve for d=1..D.
%
%  Examples
%  --------
% X = randsphere(30,1000,1);
% V = linSubspSpanOrthonormalize(randn(120,30));
% pts = V*X;
% [inds,dists] = KNN(pts,12,true);
% [d,kl] = MiND_KL(pts,'dists',dists);

    % Checking parameters:
    if nargin<1; error('A dataset is required'); end
    
    % Infos:
    [D,N] = size(data);
    
    % Default parameters:
    params = struct('k',{10}, ...
                    'normalized',{true}, ...
                    'minDim',{1}, 'maxDim',{D}, ...
                    'sigma',[]);
                
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
        [~,dists] = KNN(data,params.k+2);
    end
    
    % Normalizing the distances:
    if mustDoKnn || not(params.normalized)
        % Cropping the dists:
        dists = dists(:,2:end);
        
        % Normalizing and removing the last element:
        dists = dists(:,1:end-1)./repmat(dists(:,end),[1,size(dists,2)-1]);
    end
    
    % Only the first normalized neighbour is required:
    k = size(dists,2);
    dists = dists(:,1)';

    % Executing the experiments with unit uniformly-sampled hyper-spheres:
    numDims = params.maxDim - params.minDim + 1;
    kl = zeros(1,numDims);
    for i=1:numDims
        % Generating the data:
        data2 = randsphere(params.minDim + i - 1, N);
        
        % Computing the normalized knn distances:
        [~,distsExp] = KNN(data2,k,true);
        distsExp = distsExp(:,1)';
        
        % Estimating the Kullback-Leibler divergence:
        kl(i) = wangKL1d(distsExp,dists);
    end
    
    % Smoothing if required:
    if not(isempty(params.sigma))
        % Preparing the impulse response:
        rad = 4*ceil(params.sigma/2);
        h = gauss(-rad:rad,params.sigma,0,false);
        h = h/sum(h);
        
        % Filtering:
        kl = conv([flipdim(kl,2),kl,kl],h,'same');
        kl = kl(numDims+1:end-numDims);
    end

    % Estimating the id:
    [~,d] = min(kl);
    d = d + params.minDim - 1;
end

% ------------------------ LOCAL FUNCTIONS ------------------------

% Wang's Kullback-Leibler divergence estimator in one dimension:
function div = wangKL1d(data1,data2)
    % Getting data:
    n = numel(data1);
    m = numel(data2);
    data1 = sort(data1);
    
    % Generating the nearest neighbors for data1:
    nnData1 = [-inf,data1,inf];
    nnData1 = abs(nnData1(1:end-1)-nnData1(2:end));
    nnData1 = min([nnData1(1:end-1);nnData1(2:end)],[],1);
    
    % Generating the nearest neighbors for data2:
    nnData2 = min(abs(repmat(data1,[m,1]) - repmat(data2',[1,n])),[],1);
    
    % Computing the kl div:
    div = abs(sum(log(nnData2./(nnData1+eps)))/n + log(m/(n-1)));
end


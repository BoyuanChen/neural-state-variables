function [isLowDim,dHat,mu,tau] = DANCo_statistics(data,k,params,force)
%DANCO_STATISTICS Computes the statistics used by DANCo
%
% function [isLowDim,dHat,mu,tau] = DANCo_statistics(data,k,params,force)
%
%  This funciton allows to compute the statistic parameters used by DANCo
% to estimate the intrinsic dimensionality, in cas of isLowDim==true only
% the dHat value is computed (the other output values set to 0 in this
% case).
%
%  Parameters
%  ----------
% IN:
%  data     = The dataset as a matrix with columnwise vectors.
%  k        = KNN parameter.
%  params   = The named parameters passed to DANCo (see DANCo or DANCoFit).
%  force    = Forces the computation of mu and tau. (def=false).
% OUT:
%  isLowDim =  True if the intrinsic dimensionality is considered low, in
%             this case the dHat estimation is considered good enough.
%  dHat     = Estimated id by means of biased esimators like MLE or MiND.
%  mu,tau   = Von Mises circular statistic parameters.

    % Checking parameters:
    if nargin<3; error('A dataset, k, and the parameters are required'); end
    if nargin<4; force=false; end
    if not(isa(params,'struct')); error('Params must be a struct'); end
    if not(isfield(params,'mlMethod')); error('An mlMethod is required'); end
    if not(isfield(params,'minCorrectionDim')); error('A minCorrectionDim is required'); end
    
    % Infos:
    N = size(data,2);
    
    % Is the knn already there?
    mustDoKnn = true;
    if isfield(params,'inds') && isfield(params,'dists')
        % Extracting and checking:
        inds = params.inds;
        dists = params.dists;
        if size(dists,1) == N && size(dists,2) >= k+2 && size(inds,1) == N && size(inds,2) >= k+2
            % KNN already present:
            dists = dists(:,1:k+2);
            inds = inds(:,1:k+2);
            mustDoKnn = false;
        end
    end
    
    % Finding the KNNs only if needed:
    if mustDoKnn
        % It's needed!
        [inds,dists] = KNN(data,k,true);
    end
    
    % Normalizing the distances:
    if not(mustDoKnn || (isfield(params,'normalized') && params.normalized))
        % Cropping both dists and inds:
        dists = dists(:,2:end);
        inds = inds(:,2:end);
        
        % Normalizing and removing the last element:
        dists = dists(:,1:end-1)./repmat(dists(:,end),[1,size(dists,2)-1]);
    end
    
    % Estimating dHat:
    switch params.mlMethod
        case 'MLE'
            dHat = MLE(data,'dists',dists,'normalized',true);
        case 'MiND_MLI'
            dHat = MiND_ML(data,'dists',dists,'optimize',false,'normalized',true);
        case 'MiND_MLK'
            dHat = MiND_ML(data,'dists',dists,'optimize',true,'normalized',true);
        otherwise
            error(['Unknown mlMethod: ' params.mlMethod]);
    end
    
    % Correction based on the KL-divergence:
    isLowDim = dHat < params.minCorrectionDim;
    if not(isLowDim) || force
        % The circular statistics:
        angles = ComputeAngles(data,inds);
        [mu,tau] = CircStats(angles);

        % Averaging the parameters:
        mu = angle(sum(exp(1i*mu)));
        tau = mean(tau);
    else
        % Fake parameters:
        mu = 0;
        tau = 0;
    end

end

% ------------------------ LOCAL FUNCTIONS ------------------------

% Computing the used angular statistics:
function [mu,tau] = CircStats(angles)
    % Computing the complex circular means:
    cm = sum(exp(1i*angles),2);
    
    % Computing the von Mises parameters:
    mu = angle(cm);
    r = abs(cm)/size(angles,2);
    
    % The R function:
    if r < 0.53; tau = 2*r + r.^3 + 5*r.^5./6; else
        if r < 0.85; tau = -0.4 + 1.39*r + 0.43./(1-r);
        else tau = 1./(r.^3 - 4*r.^2 + 3*r); end
    end
    
    % Correcting for small sample sizes:
    if size(angles,2) < 15
        mask = tau < 2;
        tau(mask) = max(tau(mask)-2./(size(angles,2).*tau(mask)),0);
        tau(~mask) = (size(angles,2)-1).^3.*tau(~mask)./(size(angles,2).^3+size(angles,2));
    end
end

% -----------------------------------------------------------------

% Computing the pairwase angles for neighbours:
function angles = ComputeAngles(data,inds)
    % Init:
    K = size(inds,2);
    angles = zeros(size(inds,1),K*(K-1)/2);
    cnk = combnk(1:size(inds,2),2);
    
    % Iterating on points:
    for i=1:size(data,2)
        % Translating and normalizing points:
        pts = data(:,inds(i,:)) - repmat(data(:,i),[1,K]);
        pts = pts./repmat(sqrt(sum(pts.^2,1)),[size(pts,1),1]);
        
        % Computing all the pairwise angles:
        a = pts(:,cnk(:,1)); b = pts(:,cnk(:,2));
        angles(i,:) = atan2(sqrt(sum((b-repmat(sum(a.*b,1),[size(a,1),1]).*a).^2,1)),sum(a.*b,1));
    end
end

function [inds,dists] = KNN(X,k,normalized)
%KNN  A multi-version knnsearch
%
% function [inds,dists] = KNN(X,k)
%
%  This function allows to identify the indexes (and optionally the
% distances) of the first k neighbours of each given columnwise point.
%
% IN:
%  X        = The dataset, each column represents a point.
%  k        = The number of neighbours of interest. (def=10)
%  normalized = Must the k neighbours be normalized? 
%           (triggers a k+2 search, def=false)
% OUT:
%  inds     = Each row i-th contains the neighbour indexes of the i-th point.
%  dists    = Each row i-th contains the neighbour distances of the i-th point.

    % Checking parameters:
    if nargin<1; error('Dataset required.'); end
    if nargin<2; k=10; end
    if nargin<3; normalized=false; end

    % Managing th normalized:
    if normalized; k=k+2; end
    
    % Checking the version:
    if datenum(version('-date')) < 734729
        % Computing all the distances:
        dists = L2(X, X);
        
        % Sorting:
        [dists, inds] = sort(dists, 2);
        
        % Choosing the first k elements:
        dists = dists(:,1:k);
        inds = inds(:,1:k);
    else
        % The Matlab implementation:
        X = X';
        [inds, dists] = knnsearch(X,X,'K',k);
    end
    
    % Managing th normalized:
    if normalized
        % Cropping both inds and dists:
        inds = inds(:,2:end-1);
        dists = dists(:,2:end);
        
        % Normalizing and removing the last element:
        dists = dists(:,1:end-1)./repmat(dists(:,end),[1,size(dists,2)-1]);
    end
end

% ------------------------ LOCAL FUNCTIONS ------------------------

% L2 distance (by Roland Bunschoten):
function d = L2(a,b)
    d = real(sqrt(bsxfun(@plus, sum(a .* a)', bsxfun(@minus, sum(b .* b), 2 * a' * b))));
end

function X = randsphere(d,N,r)
% RANDSPHERE  Random points uniformly drawn from a spere with radius r
%
% function X = randsphere(d,N,r)
%
%  This function generates random data samples from a uniform hypersphere
% with a given radius. The algorithm is the one reported by Roger Stafford
% on matlabcentral/fileexchange.
%
%  Parameters
%  ----------
% IN:
%  d    = Space dimensionality. (def=2)
%  N    = Number of points. (def=1000)
%  r    = Hypersphere radius. (def=1)
% OUT:
%  X    = Matrix of points (columnwise).

    % Checking parameters:
    if nargin<1; d=2; end
    if nargin<2; N=1000; end
    if nargin<3; r=1; end

    % Gaussian randomly sampled points:
    X = randn(d,N);
    
    % Squared norms:
    s2 = sum(X.^2,1);
    
    % Correcting the points norms:
    X = X.*repmat(r*(gammainc(s2/2,d/2).^(1/d))./sqrt(s2),d,1);
end


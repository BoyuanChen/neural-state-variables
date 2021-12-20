function Vo = linSubspSpanOrthonormalize(V)
% LINSUBSPSPANORTHONORMALIZE  Orthonormalizes a linear subspace span matrix.
%
% function Vo = linSubspSpanOrthonormalize(V)
%
%  This funciton takes a DxN matrix that represents a set of N linear
% independent vectors of a D-dimensional space, and that identifies an
% N-dimensional linear subspace, and orthonormalizes them so that the
% subspace spanned is the same but the vectors are versors orthogonal one
% each other. The solutions to this problem are infinitely many, the one
% given is the one that normalizes the first vector, orthonormalizes the
% second with respect to the plane containing the first two vectors,
% orthonormalizes the third with respect to the 3D space spanned by the
% first three vectors and so on. Obviously must be 1<=N<=D.
%
%  Parameters
%  ----------
% IN:
%  V    = The DxN matrix containing the N vectors.
% OUT:
%  Vo   = The orthonormalized linear subspace base.
%
%  Pre
%  ---
% -  The columns of V must be linear independent (i.e. rank(V')==N).
% -  The columns of V must be >=1 and <=D.
%
%  Post
%  ----
% -  The returned Vo contains columns that are normal
%   (i.e. norm(V(:,i)==1).
% -  The returned Vo contains columns that are orthogonal
%   (i.e. dot(V(:,i),V(:,j))==0 for i~=j).

% Check params:
if size(V,2)<1 || size(V,2)>size(V,1)
    error('N columns must be >=1 and <=D rows!');
end

% Check for linearly dependence:
if rank(V')~=size(V,2)
    error('The V vectors must be linearly independent!');
end

% Normalizing the V vectors:
normV = sqrt(sum(V.^2));
normV(normV==0) = eps;
V = V./repmat(normV,[size(V,1),1]);

% Orthogonalizing:
for i=2:size(V,2)
    % Computing the projection on the subspace:
    proj = V(:,1)*dot(V(:,1),V(:,i));
    for j=2:i-1
        % The projection of the Vi vector over the Vj orthonormal versor:
        proj = proj + V(:,j)*dot(V(:,j),V(:,i));
    end
    % Removing and normalizing:
    V(:,i) = V(:,i) - proj;
    V(:,i) = V(:,i)./norm(V(:,i));
end

% Returning:
Vo = V;

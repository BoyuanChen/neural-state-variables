function kl = DANCo_estimateKL(k,dHat,mu,tau,dHat_ref,mu_ref,tau_ref)
%DANCO_ESTIMATEKL Estimating the Kullback-Leibler divergence for DANCo
%
% function kl = DANCo_estimateKL(k,dHat,mu,tau,dHat_ref,mu_ref,tau_ref)
%
%  This function estimates the Kullback-Leibler divergence between the
% distances/angles bivariate distributions over two datasets whose
% statistic parameters are {dHat,mu,tau} and {dHat_ref,mu_ref,tau_ref}
% respectively.
%
%  Parameters
%  ----------
% IN:
%  k                        = The KNN parameter.
%  dHat,mu,tau              = Statistics for the real dataset.
%  dHat_ref,mu_ref,tau_ref  = Statistics for the synthetic dataset.

    % Checking parameters:
    if nargin<7; error('Too few parameters'); end

    % Doing the computation:
    kl = EstimateKL_dists(dHat,dHat_ref,k) + ...
        EstimateKL_angles(mu,tau,mu_ref,tau_ref);
    
end

% ------------------------ LOCAL FUNCTIONS ------------------------

% Estimating the KL-divergence for all the cases d=1:D:
function kl = EstimateKL_angles(m1,k1,m2,k2)
    % The bessel values:
    bk2 = besseli(0,k2);
    bk1 = besseli(0,k1);
    ak1 = besseli(1,k1);
    ck1 = besseli(1,-k1);

    % Checking for infs:
    if(isinf(k1)); k1 = realmax; end
    k2(isinf(k2)) = realmax;
    bk2(isinf(bk2)) = realmax;
    if(isinf(bk1)); bk1 = realmax; end
    if(isinf(ak1)); ak1 = realmax; end
    if(isinf(ck1)); ck1 = realmax; end
    
    % Computing the KL:
    kl =  abs(log(bk2./bk1) + (ak1-ck1)./(2*bk1).*(k1-k2.*cos(m2-m1)));
end

% -----------------------------------------------------------------

% Estimating the KL-divergence for the distances:
function kl = EstimateKL_dists(d1,d2,k)
    % Initializing:
    bin = zeros(numel(d2),k+1);
    
    % Computing the terms in the summary:
    for i=1:k+1
        bin(:,i) = (-1)^(i-1)*nchoosek(k,i-1).*psi(1+(i-1)*(d1./d2)); 
    end
    
    % Estimating the KL in cloised form:
    kl = -(1+harmonic(k-1)-harmonic(k)*(d2./d1)+log(d2./d1)+(k-1)*sum(bin,2)');
end

% -----------------------------------------------------------------

% Harmonic numbers (by David Terr):
function h = harmonic(z)
    if z == 1
        h = 1;
    else
        h = log(z) + 0.5772196649 + 1/(2*z) - 1/(12*z^2) + 1/(120*z^4) - 1/(252*z^6); 
    end
end

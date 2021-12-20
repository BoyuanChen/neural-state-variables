function func = gauss(x, sigma, mu, norm)
% GAUSS  Returns the monodimensional gaussian function
%
%  Return the evaluation of the gaussian function on an array of points
%
%  Params:
%
%  x:       The array of x points (def=[-10:10])
%  sigma:   The standard deviation (def=2.5)
%  mu:      The mean of th e gaussian (def=0)
%  norm:    Must be normalized? (def=true)

% Check x
if nargin<1
    x = -10:10;
end

% Check sigma
if nargin<2
    sigma = 2.5;
end

% Check mean
if nargin<3
    mu = 0;
end

% Check norm
if nargin<4
    norm = true;
end

% Computing the gaussian funcion
if norm
    func = exp(-(x-mu).^2/(2*sigma^2))/(sigma*sqrt(2*pi));
else
    func = exp(-(x-mu).^2/(2*sigma^2));
end

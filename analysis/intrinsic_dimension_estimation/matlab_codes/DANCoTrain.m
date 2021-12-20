function DANCoTrain(modelfile,varargin)
% DANCoTrain  Training the DANCoFit model file.
%
% function DANCoTrain(modelfile, ParamName,ParamValue, ...)
%
%  This funciton executes experiments with points uniformly drawn from
% hyper-spheres with unitary radius to produce the fitting functions used 
% by DANCoFit, if you use this please cite:
%
%  "DANCo: Dimensionality from Angle and Norm Concentration"
%  C. Ceruti, S. Bassis, A. Rozza, G. Lombardi, E. Casiraghi, P. Campadelli
%  arXiv:1206.3881v1, http://arxiv.org/abs/1206.3881v1
%
%  Parameters
%  ----------
% IN:
%  modelfile = The name of the model file to be produced.
% Named parameters:
%  'k'
%       The number of neighbours to be used, must be >= 1. (def=10)
%  'maxDim'
%        The maximal dimensionality to be used in the experiments, more
%       dims means more computation but also accurate fits. (def=100)
%  'cardinalities'
%        The list of cardinalities to be used in the experiments, more
%       cardinalities means more computation but also accurate fits, the
%       suggested choice is to adopt an 1-2-5 series. 
%       (def=[50,100,200,500,1000,2000,5000])
%  'iterations'
%        The number of iterations for experiment (results averaged), more 
%       iterations means more computation but also accurate fits. (def=100)
%  'mlMethod'
%       Algorithm used to pre-estimate d:
%               MLE         = Levina's algorithm.
%               MiND_MLI    = ML on the first neighbour.
%               MiND_MLK    = ML on the first neighbour and optimization.
%             (def='MiND_MLK')
%
%  Example
%  -------
% DANCoTrain('sampleModel','maxDim',20,'cardinalities',[200,500,1000],'iterations',1);
% X = randsphere(10,500);
% V = linSubspSpanOrthonormalize(randn(20,10));
% pts = V*X;
% DANCoFit(pts,'modelfile','sampleModel')

    % Checking parameters:
    if nargin<1; error('A model file name is required.'); end

    % Default parameters:
    params = struct('k',{10}, ...
                    'maxDim',{100}, ...
                    'cardinalities',{[50,100,200,500,1000,2000,5000]}, ...
                    'iterations',{100}, ...
                    'mlMethod',{'MiND_MLK'});
                
    % Parsing named parameters:
    if nargin > 1
        try
            params = parseParamsNamed(varargin, false, params);
        catch e
            error('Pairs ''name,value'' are required as arguments after the dataset.');
        end
    end
    
    % Parameters to be passed to the statistics computation function:
    params2 = struct( ...
        'mlMethod',{params.mlMethod}, ...
        'minCorrectionDim',{1});
    k = params.k;
    
    % Computing the size of the statistic matrices:
    N = numel(params.cardinalities);
    D = params.maxDim;
    
    % Initialization:
    dHat = zeros(D-1,N);
    mu = zeros(D-1,N);
    tau = zeros(D-1,N);
    for d = 2:D
        for n = 1:N
            % Initializing the data of this iteration:
            dHat_ref = zeros(1,params.iterations);
            mu_ref = zeros(1,params.iterations);
            tau_ref = zeros(1,params.iterations);

            % Executing the experiments:
            for i = 1:params.iterations
                % Generating the dataset:
                data = randsphere(d, params.cardinalities(n));
                
                % Sayng to the user what I'm doing:
                fprintf('(d = %d \tn = %d \ti = %d)...', ...
                    d,params.cardinalities(n),i);

                % Estimating the parameters:
                [~,dHat_ref(i),mu_ref(i),tau_ref(i)] = ...
                    DANCo_statistics(data,k,params2,true);
                
                % Sayng to the user what I'm doing:
                fprintf(' \tDONE!\n');
            end
            
            % Storing the results:
            dHat(d-1,n) = mean(dHat_ref);
            mu(d-1,n) = mean(mu_ref);
            tau(d-1,n) = mean(tau_ref);
        end
    end
    
    % Sayng to the user what I'm doing:
    fprintf('Fitting & saving...');
    
    % Initializing:
    inVars = {2:D,params.cardinalities};
                
    % Fitting the funcitons:
    fitDhatFun = fnxtr(csaps(inVars,dHat),2);
    fitMuFun = fnxtr(csaps(inVars,mu),2);
    fitTauFun = fnxtr(csaps(inVars,tau),2);
    
    % Producing function handles:
    fitDhat = @(D,N)(fnval(fitDhatFun,{D,N})');
    fitMu = @(D,N)(fnval(fitMuFun,{D,N})');
    fitTau = @(D,N)(fnval(fitTauFun,{D,N})');
    
    % Writing the model:
    save(modelfile,'fitMu','fitTau','fitDhat','k');
    
    % Sayng to the user what I'm doing:
    fprintf(' \tDONE!\n');

end

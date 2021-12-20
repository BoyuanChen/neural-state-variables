function pars = parseParamsNamed(parsCell,recurse,parsi,multi)
% PARSEPARAMSNAMED  Getting named parameters as a structure.
%
% function pars = parseParamsNamed(parsCell,recurse,parsi,multi)
%
%  This function allows to get named parameters as a struct object. Named
% parameters are sequences of strings and values pair where the string
% represents the name of the parameter and the value its value.
%
%  Parameters
%  ----------
% IN:
%  parsCell     = The cell containing the parameters (like varargin).
%  recurse      = Must the parsing recurse in sub-cells? (def=false)
%  parsi        = Initial struct (with default values). (def=struct)
%  multi        = Must the multi-valued fields be grouped in a cell? (def=false)
% OUT:
%  pars         = The structure containing the parameters.

% Chck parameters:
if nargin<1 || ~isa(parsCell,'cell')
    error('A cell with parameters must be passed as parameter!');
end
if nargin<2; recurse=false; end
if nargin<3; parsi=struct; end
if nargin<4; multi=false; end

% Check for the size:
N = numel(parsCell);
if mod(N,2)~=0
    error('The cell must contain an even number of elements!');
end

% Init:
pars = parsi;

% Iterating on parameters:
for i=1:2:N
    % Get the name:
    name = parsCell{i};
    if ~isa(name,'char')
        error('The name of parameters must be a string!');
    end
    
    % Getting the value:
    value = parsCell{i+1};
    if recurse && isa(value,'cell')
        % Generating the substructure:
        value = parseParamsNamed(value,recurse);
    end
    
    % Checking if multi:
    if multi
        % Init:
        if not(isfield(pars,name))
            pars.(name) = {};
        end
        if not(isa(pars.(name),'cell'))
            pars.(name) = {pars.(name)};
        end
        
        % Packing:
        pars.(name){end+1} = value;
    else
        % Saving the last:
        pars.(name) = value;
    end
end

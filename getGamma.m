function [gamma g]=getGamma(x,c,varargin)
% Initialize by defaut the gamma parameter of PFCM
%    [gamma g]=getGamma(x,c,varargin)
%
% INPUTS
%   x: input matrix nxd
%   c: number of desired clusters
%   varargin: optional arguments
%   - 'distance': 'sqEuclidean' (default) or 'Mahalanobis'
%   - 'm': coefficient controling the fuzziness of the probabilistic
%          partition
%
% OUTPUTS
%   gamma: vector (cx1) corresponding to the gamma to use for each cluster
%   g: matrix cxd corresponding to the centers of the clusters
%
% Reference:
% [1] R. Krishnapuram, J. Keller. "A possibilistic approach to clustering", 
% IEEE Trans. Fuzzy Syst. 1 (1993) 98–110.
%
% Remarks:
% -> K the weighting factor enabling to reduce or increase the overall
% size of the clusters is set to 1 by default.
%
%  --------------------------------------------------------------------------
% Author : Violaine Antoine
% mail   : violaine.antoine@uca.fr
% date   : 09-07-2018
% version: 1.0
    
%%%%%%%%%%%% OPTIONAL PARAMETERS %%%%%%%%%%%%
  ip=inputParser;
  defaultDistance='sqEuclidean';
  defaultm=2;

  expectedDistances = {'sqEuclidean','Mahalanobis'};
  
  addOptional(ip,'distance',defaultDistance,@(x) any(validatestring(x,expectedDistances)));
  addOptional(ip,'m',defaultm,@(x) x>0);

  ip.parse(varargin{:});
  dist=ip.Results.distance;
  m=ip.Results.m;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  [n nbAtt]=size(x);
  
  % run pfcm
  [P,U,g,S]=pfcm(x,c,'a',1,'b',0,'m',m,'gamma',zeros(1,c),'distance',dist);

  % compute distances
  if isempty(S)
    for k=1:c
      S=[S {eye(nbAtt)}];
    end
  end
  
  for k=1:c
    aux=x-repmat(g(k,:),n,1);
    D(:,k)=det(S{k})^(1/nbAtt)*dot(aux*inv(S{k}),aux,2);
  end
  
  % compute gamma
  gamma=sum(((U.^m).*D))./sum(U);
  % K=1
  % gamma=K*(sum(((U.^m).*D))./sum(U));
function [D,Sellipse] = pfcm_distCalculation(x,g,varargin);
% Distance Calculation for pfcm and its variants
%    [D,Sellipse] = pfcm_distCalculation(x,g,varargin)
%
% INPUTS
%   x: input matrix nxd
%   g: matrix Kxd corresponding to the centers of the clusters
%   varargin: optional arguments
%   - 'distance': 'sqEuclidean' (default) or 'Mahalanobis'
%   - 'rho' : 1xc vector for the volume of the ellipse (1 vector by default)
%   - 'beta' : coefficient controling the fuzziness of the final partition
%              (2 by default)
%   - 'partition': actual possibilistic partition of the dataset. 
%                  Usefull only with a Mahalanobis distance
%
% OUTPUTS
%   D: Distance matrix nxK between points and clusters
%   Sellipse: vector 1xK of covariance matrices (nbAttxnbAtt) in case of a Mahalanobis distance.
%             (empty vector for a Euclidean distance)
%
% References:
%
% [1] E. Gustafson and W. Kessel. Fuzzy clustering with a fuzzy covariance 
%     matrix. Proceedings of the IEEE CDC. San Diego, California, USA, 
%     pp. 761–766, 1979.
% [2] R. Babuska and all. Improved Covariance Estimation for Gustafson-Kessel
%     Clustering. Honolulu, Hawaii, FUZZ-IEEE. pp. 1081-1085, 2002.
% [3] B. Ojeda-Magaina, R. Ruelas, M. Corona-Nakamura, and D. Andina. An 
%     improvement to the possibilistic fuzzy c-means clustering algorithm. 
%     WAC’06, 2006.
%  --------------------------------------------------------------------------
% Author : Violaine Antoine
% mail   : violaine.antoine@uca.fr
% date   : 07-27-2017
% version: 1.1

[c nbAtt]=size(g);
[n nbAtt]=size(x);

%%%%%%%%%%%% OPTIONAL PARAMETERS %%%%%%%%%%%%
ip=inputParser;
defaultDistance='sqEuclidean';
defaultRho=ones(1,c);
expectedDistances = {'sqEuclidean','Mahalanobis'};
defaultDebug=0;

addOptional(ip,'rho',defaultRho,@(x) isequal(size(x),[1 c]));
addOptional(ip,'distance',defaultDistance,@(x) any(validatestring(x,expectedDistances)));
addOptional(ip,'partition',@isnumeric);
addOptional(ip,'debug',defaultDebug,@(x) x==1 | x==0);

ip.parse(varargin{:});
rho=ip.Results.rho;
dbug=ip.Results.debug;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Sellipse=[];
if strcmp(ip.Results.distance,'sqEuclidean') % Euclidean distance
  D=bsxfun(@plus,dot(x,x,2),dot(g,g,2)')-2*(x*g');
else % mahalanobis distance

  P=ip.Results.partition;

  D=zeros(n,c);
  for k=1:c
    % calcul of Ck
    aux=x-repmat(g(k,:),n,1);
    Ck=aux'*(repmat(P(:,k),1,nbAtt).*aux);
    Ck=Ck/sum(P(:,k));
    Sellipse=[Sellipse {Ck}];

    %calcul of distance
    if rcond(Ck)>10^-15
       D(:,k)=rho(k)*det(Ck)^(1/nbAtt)*dot(aux*inv(Ck),aux,2);
    else % if the matrix is singular, use of formula in [2]

      % parameters for the estimation
      gamma=eps;
      beta=10^15;

      % compute of the estimation
      Fk=(1-gamma)*Ck+gamma*rho(k)*det(Ck)^(1/nbAtt)*eye(nbAtt);
      Fk=(Fk+Fk')/2;  % numerical correction of Fk for the symetrie.
                       % if not, the eigenvalues can be images
 
      [eigVect eigVal]=eig(Fk);
      eigVal=diag(eigVal);
      eigValMax=max(eigVal);
      eigVal(eigValMax>beta*eigVal)=eigValMax/beta;
      Ck=eigVect*diag(eigVal)*inv(eigVect);
      Sellipse{k}=Ck;
      
      D(:,end)=diag(det(Ck)^(1/nbAtt)*aux*inv(Ck)*aux');
    end
  end
  
end

function [D,Sellipse] = spfcm_distCalculation(x,g,varargin);
% Distance Calculation for spfcm
%    [D,Sellipse] = spfcm_distCalculation(x,g,varargin)
%
% INPUTS
%   x: input matrix nxd
%   g: matrix Kxd corresponding to the centers of the clusters
%   varargin: optional arguments
%   - 'distance': 'sqEuclidean' (default) or 'Mahalanobis'
%   - 'beta' : coefficient controling the fuzziness of the final partition
%              (2 by default)
%   - 'partition': actual possibilistic partition of the dataset. 
%                  Usefull only with a Mahalanobis distance
%   - 'rho' : 1xc vector for the volume of the ellipse (1 vector by default)
%
% OUTPUTS
%   D: Distance matrix nxK between points and clusters
%   Sellipse: vector 1xK of covariance matrices (nbAttxnbAtt) in case of a Mahalanobis distance.
%             (empty vector for a Euclidean distance)
%
%  --------------------------------------------------------------------------
% Author : Violaine Antoine
% mail   : violaine.antoine@uca.fr
% date   : 08-10-2022
% version: 1.2

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

errSing=0;
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
    if rcond(Ck)>10^-15 % chek the singularity of the matrix (det(Ck)
                        % also but it has numerical issue with matlab)
      D(:,k)=rho(k)*real(det(Ck)^(1/nbAtt))*dot(aux*inv(Ck),aux,2);

    else 
      errSing=1;
    end
  end
end

if errSing
  D=[];
end
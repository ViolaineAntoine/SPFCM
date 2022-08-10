function [T,U,g,Sellipse,J,t,errMin] = pfcm(x,c,varargin)
% Possibilistic Fuzzy C-Means 
%    [P,U,g,Sellipse,J,t] = pfcm(x,K,varargin)
%
% INPUTS
%   x: input matrix nxd
%   K: number of desired clusters
%   varargin: optional arguments
%   - 'distance': 'sqEuclidean' (default) or 'Mahalanobis'
%   - 'rho' : 1xc vector for the volume of the ellipse (1 vector by default)
%   - 'ginit': matrix Kxd corresponding to the initial centers of the clusters
%              (random initialization by default)
%   - 'm' : coefficient controling the fuzziness of the probabilistic partition
%           (2 by default)
%   - 'a': coefficient >0 giving importance to probability memberships (1
%          by default)
%   - 'b': coefficient >0 giving importance to typicality memberships (1
%          by default) 
%   - 'eta': coefficients controling the fuzziness of the typicality values
%            (2 by default)
%   - 'gamma' : vector (1xc) of coefficients controling the Krishnapuram and 
%               Keller term.
%   - 'debug': check that the objectif function is well-minimized (0 by default)
%
% OUTPUTS
%   T: possibilistic partition (equal to typicality values)
%   U: fuzzy partition
%   g: matrix Kxd corresponding to the centers of the clusters
%   Sellipse: covariance matrices Kx(pxp) if existing (Mahalanobis case)
%   J: result of the objective function
%   t: iterations made before convergence
%   errMin: error in the minimization (debug option)
%
% Reference:
% [1]  N. Pal, K. Pal, J. Keller, and J. Bezdek. A possibilistic fuzzy
% c-means clustering algorithm. IEEE Transactions on Fuzzy Systems, 2005.
% [2] B. Ojeda-Magana, R. Ruelas, M. Corona-Nakamura, and D. Andina. An
%     improvement to the possibilistic fuzzy c-means clustering algorithm. 
%     World Automatic Control Conference  (WAC06), 2006.
%
%  --------------------------------------------------------------------------
% Author : Violaine Antoine
% mail   : violaine.antoine@uca.fr
% date   : 07-26-2017
% version: 1.0

%%%%%%%%%%%% OPTIONAL PARAMETERS %%%%%%%%%%%%
[n,nbAtt]=size(x);

ip=inputParser;
defaultDistance='sqEuclidean';
defaultRho=ones(1,c);
defaultGinit=rand(c,nbAtt).*repmat(max(x)-min(x),c,1)+repmat(min(x),c,1);
defaulta=1;
defaultb=1;
defaultm=2;
defaulteta=2;
defaultgamma=ones(1,c);
defaultDebug=0;

expectedDistances = {'sqEuclidean','Mahalanobis'};

addOptional(ip,'distance',defaultDistance,@(x) any(validatestring(x,expectedDistances)));
addOptional(ip,'rho',defaultRho,@(x) isequal(size(x),[1 c]));
addOptional(ip,'ginit',defaultGinit,@(x) isequal(size(x),[c nbAtt]));
addOptional(ip,'a',defaulta,@(x) x>=0);
addOptional(ip,'b',defaultb,@(x) x>=0);
addOptional(ip,'m',defaultm,@(x) x>0);
addOptional(ip,'eta',defaulteta,@(x) x>0);
addOptional(ip,'gamma',defaultgamma,@(x) isequal(size(x),[1 c]));
addOptional(ip,'debug',defaultDebug,@(x) x==1 | x==0);

ip.parse(varargin{:});
dist=ip.Results.distance;
rho=ip.Results.rho;
a=ip.Results.a;
b=ip.Results.b;
m=ip.Results.m;
eta=ip.Results.eta;
gamma=ip.Results.gamma;
dbug=ip.Results.debug;

maxIter=10^5;
epsilon=10^-3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

epsPrototypes=1e-3;

%%%%%%%%%%%%% Initializations %%%%%%%%%%%%% 
% -- initialization of memberships
g=ip.Results.ginit;
[D,~]=pfcm_distCalculation(x,g);
U=(1./D)./repmat(sum(1./D,2),1,c);
if b>0
Taux=((b*D)./repmat(gamma,n,1)).^(eta-1)+1;
T=1./Taux;
else
T=zeros(n,c);
end

%% ------------------------ iterations--------------------------------
t=0;  % iteration

errMin=0;
notFinished=1;
Jold=Inf;

while notFinished

  % Update prototypes (equation 23 from [1])
  gold=g;
  PU=a*U.^m+b*T.^eta;
  for k=1:c
    g(k,:)=sum(x.*repmat(PU(:,k),1,nbAtt))/sum(PU(:,k));
  end
   %g(isnan(g))=0;

%aux=x-repmat(g(k,:),n,1);
%for k=1:c
%  Daux(:,k)=det(Sellipse{k})^(1/nbAtt)*dot(aux*inv(Sellipse{k}),aux,2);
%end

  % Update distance (variable D=D_ik)
  [D,Sellipse]=pfcm_distCalculation(x,g,'distance',dist,'partition',PU,'rho',rho);
  J1=pfcm_function(U,T,D,a,b,m,eta,gamma);
  
  % compute memberships U (equation 21 from [1])
  auxU=bsxfun(@times,D,sum(1./D,2)).^(1/(m-1));
  UOld=U;
  U=1./auxU;
  U(isnan(U))=1;
  
  J2=pfcm_function(U,T,D,a,b,m,eta,gamma);
  
  % Update Typicality values (equation 22 from [1])
  if b>0
    Taux=((b*D)./repmat(gamma,n,1)).^(eta-1)+1;
    T=1./Taux;
  end % T remains 0
  
  J3=pfcm_function(U,T,D,a,b,m,eta,gamma);
   
  % Debug option : check the good minimization 
  if dbug 
    if (Jold-J1+epsilon<0 | J1+epsilon<J2 | J2+epsilon<J3)
      fprintf(1,'error of minimization in PFCM: J0=%f J1=%f J2=%f J3=%f\n',Jold,J1,J2,J3);
      keyboard;
      errMin=1;
    else
      Jold=J3;
    end
  end

  %plottest(T,g,Sellipse);


  % Prototype stabilization
  notFinished=sum(sum((abs(g-gold)>epsPrototypes))) & t<=maxIter & ~errMin;

  t=t+1; % iteration

end

J=pfcm_function(U,T,D,a,b,m,eta,gamma);

% pfcm objectif function
function J=pfcm_function(U,T,D,a,b,m,eta,gamma)
  PU=a*U.^m+b*T.^eta;
  J=sum(sum(PU.*D))+sum(gamma.*sum((1-T).^eta));

  
function plottest(U,g,S)
  load fisheriris
  x=meas;
  y=strcmp('setosa',species)*1 + strcmp('versicolor',species)*2 + strcmp('virginica',species)*3;
  
  plotCFCMSeeds(x,y,U,'centers',g,'ellipses',S);

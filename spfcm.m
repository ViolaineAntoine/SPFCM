function [T,U,g,Sellipse,J,t,errMin] = spfcm(x,c,Pconst,varargin)
% Seed Possibilistic Fuzzy C-Means 
%    [P,U,g,Sellipse,J,t,errMin] = spfcm(x,c,Pconst,varargin)
%
% INPUTS
%   x: input matrix nxd
%   c: number of desired clusters
%   Pconst: constraints yx(c+1) the first column is indices, the other columns
%          are the desired partition. A -1 value means that the constraint
%          for the particular cluster should not be taken in account.
%   varargin: optional arguments
%   - 'distance': 'sqEuclidean' (default) or 'Mahalanobis'
%   - 'rho' : 1xc vector for the volume of the ellipse (1 vector by default)
%   - 'ginit': matrix cxd corresponding to the initial centers of the clusters
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
%   - 'coefPenaltyTerm' : coefficient >0 that penalize the non respect of the 
%                         constraints. 0 corresponds to pfcm. (1 by default)
%   - 'debug': check that the objectif function is well-minimized (0 by default)
%
% OUTPUTS
%   T: possibilistic partition (equal to typicality values)
%   U: fuzzy partition
%   g: matrix cxd corresponding to the centers of the clusters
%   Sellipse: covariance matrices cx(pxp) if existing (Mahalanobis case)
%   J: result of the objective function
%   t: iterations made before convergence
%   errMin: error in the minimization (debug option)
%
% Reference:
% [1] N. Pal, K. Pal, J. Keller, J. Bezdek. "A possibilistic fuzzy
% c-means clustering algorithm", IEEE transactions on fuzzy systems, (13)
% 4 pp 517-530 2005.
% [2] V. Antoine, J. Guerrero, G. Romero. "Possibilistic fuzzy c-means
% with partial supervision", Fuzzy Set and Systems, 2022.
% [3] Yi Cao (2022). Munkres Assignment Algorithm 
% (https://www.mathworks.com/matlabcentral/fileexchange/20328-munkres-assignment-algorithm), 
% MATLAB Central File Exchange. Retrieved August 10, 2022.
%
% Remarks:
% -> m and eta have been set to 2.
% -> a mapping function has been add to avoid degenerate solution
%
%  --------------------------------------------------------------------------
% Author : Violaine Antoine
% mail   : violaine.antoine@uca.fr
% date   : 08-10-2022
% version: 1.0

%%%%%%%%%%%% OPTIONAL PARAMETERS %%%%%%%%%%%%
[n,nbAtt]=size(x);

ip=inputParser;
defaultDistance='sqEuclidean';
defaultRho=ones(1,c);
defaultGinit=rand(c,nbAtt).*repmat(max(x)-min(x),c,1)+repmat(min(x),c,1);
defaulta=1;
defaultb=1;
defaultgamma=ones(1,c);
defaultCoef=1;
defaultDebug=0;

expectedDistances = {'sqEuclidean','Mahalanobis'};

addOptional(ip,'distance',defaultDistance,@(x) any(validatestring(x,expectedDistances)));
addOptional(ip,'rho',defaultRho,@(x) isequal(size(x),[1 c]));
addOptional(ip,'ginit',defaultGinit,@(x) isequal(size(x),[c nbAtt]));
addOptional(ip,'a',defaulta,@(x) x>=0);
addOptional(ip,'b',defaultb,@(x) x>=0);
addOptional(ip,'gamma',defaultgamma,@(x) isequal(size(x),[1 c]));
addOptional(ip,'coefPenaltyTerm',defaultCoef,@(x) x>=0);
addOptional(ip,'debug',defaultDebug,@(x) x==1 | x==0);

ip.parse(varargin{:});
dist=ip.Results.distance;
rho=ip.Results.rho;
a=ip.Results.a;
b=ip.Results.b;
gamma=ip.Results.gamma;
coef=ip.Results.coefPenaltyTerm;
dbug=ip.Results.debug;

eta=2;
m=2;
maxIter=10^4;
epsilon=10^-3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% Constraints transformation %%%%%%%%%%%%
% setting constraints matrix
if isempty(Pconst)
  Pconst=zeros(0,c+1);
end
indConst=Pconst(:,1);
PconstTotal=ones(n,c)*-1;
PconstTotal(indConst,:)=Pconst(:,2:end);
Bconst=(PconstTotal>=0);

%%%%%%%%%%%%% Initializations %%%%%%%%%%%%% 
% -- initialization of memberships
g=ip.Results.ginit;
[D,Sellipse]=spfcm_distCalculation(x,g);
U=(1./D)./repmat(sum(1./D,2),1,c);
if b>0
Taux=((b*D)./repmat(gamma,n,1)).^(eta-1)+1;
T=1./Taux;
else
T=zeros(n,K);    
end
%------------------------ iterations--------------------------------
t=0;  % iteration

notFinished=1;
Jold=Inf;
errMin=0;

while notFinished
  % Update prototypes
  gold=g;
  aux=a*U.^m+b*T.^eta+coef*Bconst.*(T-PconstTotal).^eta;
  for k=1:c
    g(k,:)=sum(x.*repmat(aux(:,k),1,nbAtt))/sum(aux(:,k));
  end


  % Update distance
  Dold=D;SellipseOld=Sellipse;
  part=a*U.^m+b*T.^eta+coef*Bconst.*(T-PconstTotal).^eta;
  [D,Sellipse]=spfcm_distCalculation(x,g,'distance',dist,'partition',part,'rho',rho);
  if isempty(D) % matrice semidefinie positif pbm
    D=Dold;
    Sellipse=Sellipse;
  end
  
  J1=spfcm_function(U,T,D,a,b,m,eta,gamma,coef,Bconst,PconstTotal);  
  if J1<0
    disp('erreur negatif');
    keyboard;
  end

  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % compute memberships U  
  auxU=bsxfun(@times,D,sum(1./D,2)).^(1/(m-1));
  UOld=U;
  U=1./auxU;
  U(isnan(U))=1;
  
  J2=spfcm_function(U,T,D,a,b,m,eta,gamma,coef,Bconst,PconstTotal);
  if J2<0
    disp('erreur negatif');
    keyboard;
  end
  
  % Update Typicality values
  if b>0
    Tnom=repmat(gamma,n,1)+coef*Bconst.*D.*PconstTotal;
    Tdenom=b*D+repmat(gamma,n,1)+coef*Bconst.*D;
    T=Tnom./Tdenom;
  end % T remains 0
  J3=spfcm_function(U,T,D,a,b,m,eta,gamma,coef,Bconst,PconstTotal);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if J3<0
    disp('erreur negatif');
    keyboard;
  end

  % Mapping step, if necessary
  %[PconstTotal,T,Mmin] = spfcm_mapping(U,T,D,a,b,m,eta,gamma,coef,PconstTotal);
  [PconstTotal,T,Mmin] = spfcm_hungarianMapping(U,T,D,a,b,m,eta,gamma,coef,PconstTotal);
  Bconst=(PconstTotal>=0);
  J4=spfcm_function(U,T,D,a,b,m,eta,gamma,coef,Bconst,PconstTotal);
  if J4<0
    disp('erreur negatif');
    keyboard;
  end

  % Debug option : check the good minimization 
  if dbug 
    if (Jold-J1+epsilon<0 | J1+epsilon<J2 | J2+epsilon<J3 | J3+epsilon<J4)
      fprintf(1,'error of minimization in SPFCM: J0=%f J1=%f J2=%f J3=%f J4=%f\n',Jold,J1,J2,J3,J4);
      keyboard;
      errMin=1;
    else
      Jold=J4;
    end
  end

  % Prototype stabilization
  notFinished=sum(sum((abs(g-gold)>epsilon))) & t<=maxIter & ~errMin;

  t=t+1; % iteration
end

if dbug & t>maxIter
  disp('too much iterations, optimization stopped');
end

J=spfcm_function(U,T,D,a,b,m,eta,gamma,coef,Bconst,PconstTotal);

% spfcm objectif function
function J=spfcm_function(U,T,D,a,b,m,eta,gamma,coef,Bconst,PconstTotal)
  PU=a*U.^m+b*T.^eta;
  penalty=sum(sum(Bconst.*(T-PconstTotal).^eta.*D));
  J=sum(sum(PU.*D))+sum(gamma.*sum((1-T).^eta))+coef*penalty;

  
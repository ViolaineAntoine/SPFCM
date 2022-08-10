function [PconstTotal,T,Mmin] = spfcm_hungarianMapping(U,T,D,a,b,m,eta,gamma,coef,PconstTotal)
% Mapping constraints with the current partition for spfcm
% using hungarian algorithm
%    [UconstTotal,U,Mmin] = spfcm_hungarianMapping(U,D,UconstTotal)
%
% INPUTS
%   U: fuzzy partition
%   T: possibilistic partition (typicality values)
%   D: distance between objects and centroids
%   a: coefficient >0 giving importance to probability memberships
%   b: coefficient >0 giving importance to typicality memberships
%   m: coefficient controling the fuzziness of the probabilistic
%   partition
%   eta: coefficients controling the fuzziness of the typicality values
%   gamma: vector (1xc) of coefficients controling the Krishnapuram and 
%          Keller term.
%   coef: coefficient >0 that penalize the non respect of the constraints.
%   PconstTotal: matrix (nxc) containing the constraints
%
% OUTPUTS
%   PconstTotal: matrix mapped (nxc) containing the constraints
%   T: possibilistic partition mapped
%   Mmin: new value of the objectif function
%
%  --------------------------------------------------------------------------
% Author : Violaine Antoine
% mail   : violaine.antoine@uca.fr
% date   : 09-07-2018
% version: 1.0

  [n,c]=size(T);
  Bconst=(PconstTotal>=0);
  
  % compute cost matrix
  C=zeros(c,c);
  for k=1:c
    for l=1:c
       C(k,l)=penalty_function(T(:,k),D(:,k),eta,Bconst(:,l),PconstTotal(:,l));
    end
  end

  [assignment,cost] = munkres(C);

  Mmin=zeros(c);
  Mmin(((1:c)-1)*c+assignment)=1;
  
  Jold=spfcm_function(U,T,D,a,b,m,eta,gamma,coef,Bconst,PconstTotal);
  
  if sum(assignment-(1:c)==0)>0 % label of constraints objects should be readjust
    PconstTotal=PconstTotal*Mmin;
    
    % compute new T
    Tnom=repmat(gamma,n,1)+coef*Bconst.*D.*PconstTotal;
    Tdenom=b*D+repmat(gamma,n,1)+coef*Bconst.*D;
    T=Tnom./Tdenom;
  end

  Jnew=spfcm_function(U,T,D,a,b,m,eta,gamma,coef,Bconst,PconstTotal);

  
function Js=penalty_function(T,D,eta,Bconst,PconstTotal)
  Js=sum(sum(Bconst.*(T-PconstTotal).^eta.*D));

% spfcm objectif function
function J=spfcm_function(U,T,D,a,b,m,eta,gamma,coef,Bconst,PconstTotal)
  PU=a*U.^m+b*T.^eta;
  penalty=sum(sum(Bconst.*(T-PconstTotal).^eta.*D));
  J=sum(sum(PU.*D))+sum(gamma.*sum((1-T).^eta))+coef*penalty;

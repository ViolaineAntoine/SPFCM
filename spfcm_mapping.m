function [PconstTotal,T,Mmin] = spfcm_mapping(U,T,D,a,b,m,eta,gamma,coef,PconstTotal)
% Mapping constraints with the current partition for spfcm
%    [UconstTotal,U,Mmin] = spfcm_mapping(U,D,UconstTotal)
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
% date   : 08-05-2017
% version: 1.0

[n,K]=size(U);
M0=eye(K);
Bconst=(PconstTotal>=0);

% current value of the objective function
Jmin=spfcm_function(U,T,D,a,b,m,eta,gamma,coef,Bconst,PconstTotal);
Mmin=M0;

iperm=perms(K:-1:1);
iperm=iperm(2:end,:); % delete order already existing

for i=1:1:size(iperm,1)
  M=M0(iperm(i,:)',:); % map matrix
  
  % compute new T with permutation
  if b>0
    Pct=PconstTotal*M;
    Bc=Pct>=0;
    Tnom=repmat(gamma,n,1)+coef*Bc.*D.*Pct;
    Tdenom=b*D+repmat(gamma,n,1)+coef*Bc.*D;
    Tnew=Tnom./Tdenom;
  else % T remains 0
    Tnew=T;
  end 


  % computation of the new objective function
  Jnew=spfcm_function(U,Tnew,D,a,b,m,eta,gamma,coef,Bc,Pct);
  
  if Jnew<Jmin
    Jmin=Jnew;
    Mmin=M;
  end
end

if ~isequal(Mmin,M0) % label of constraints objects should be readjust
    
  PconstTotal=PconstTotal*Mmin;

  % compute new T
  Tnom=repmat(gamma,n,1)+coef*Bconst.*D.*PconstTotal;
  Tdenom=b*D+repmat(gamma,n,1)+coef*Bconst.*D;
  T=Tnom./Tdenom;

end



% pfcm objectif function
function J=spfcm_function(U,T,D,a,b,m,eta,gamma,coef,Bconst,PconstTotal)
  PU=a*U.^m+b*T.^eta;
  penalty=sum(sum(Bconst.*(T-PconstTotal).^eta.*D));
  J=sum(sum(PU.*D))+sum(gamma.*sum((1-T).^eta))+coef*penalty;

clear
addpath(genpath('.'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Generate dataset %%%%
randn('seed',5);
n=150;

mu = [5 5];
SIGMA = [2 0; 0 2];
r1 = mvnrnd(mu,SIGMA,n);
  
mu = [10 5];
SIGMA = [2 0; 0 2];
r2 = mvnrnd(mu,SIGMA,n);
  
mu = [5 10];
SIGMA = [2 0; 0 2];
r3 = mvnrnd(mu,SIGMA,n);
  
mu = [10 10];
SIGMA = [2 0; 0 2];
r4 = mvnrnd(mu,SIGMA,n);

x=[r1; r2; r3; r4];
y=[ones(n,1); ones(n,1)*2; ones(n,1); ones(n,1)*2];  

c=2;
b=5;

%%%% Show dataset %%%%
figure();hold on;
idx=find(y==1);plot(x(idx,1),x(idx,2),'kx','markersize',8);
idx=find(y==2);plot(x(idx,1),x(idx,2),'ko','markersize',8);
title('Original Data');
axis equal tight

%%%% Run PFCM with a Euclidean distance %%%%

[gamma g0]=getGamma(x,c,'distance','sqEuclidean'); % Find gamma parameter
[P,U,g,S,J,t,errMin] = pfcm(x,c,'gamma',gamma,'a',0.5,'b',b,'distance','sqEuclidean');

clust=sign(P-repmat(max(P')',1,c))+1; % get hard partition
clust=clust(:,2)*2+clust(:,1);

figure();hold on; % visualization
idx=find(clust==1);plot(x(idx,1),x(idx,2),'kx','markersize',8);
idx=find(clust==2);plot(x(idx,1),x(idx,2),'ko','markersize',8);
title('PFCM-eucl hard partition');
axis equal tight

%%%% Run PFCM with a Mahalanobis distance %%%%

[gamma g0]=getGamma(x,c,'distance','Mahalanobis'); % Find gamma parameter

Jmin=Inf; % Run 10 times PFCM, take the minimum value
for iexp=1:10
    [P,U,g,S,J,t,errMin] = pfcm(x,c,'gamma',gamma,'a',0.5,'b',b,'distance','Mahalanobis');
    if Jmin>J
        Jmin=J;
        Pmin=P;
    end
end
P=Pmin;

clust=sign(P-repmat(max(P')',1,c))+1; % get hard partition
clust=clust(:,2)*2+clust(:,1);

figure();hold on; % visualization
idx=find(clust==1);plot(x(idx,1),x(idx,2),'kx','markersize',8);
idx=find(clust==2);plot(x(idx,1),x(idx,2),'ko','markersize',8);
title('PFCM-mah hard partition');
axis equal tight


%%%% Run SPFCM with a Mahalanobis distance %%%%

% Create 10 constraints
nbConst=10;
s = RandStream('mt19937ar','Seed',2);
idconst=randperm(s,length(y));
idconst=idconst(1:nbConst);
pconst=zeros(nbConst,c);
for k=1:c
  pconst(find(y(idconst)==k),k)=1;
end
const=[idconst' pconst];


Jmin=Inf; % Run 10 times SPFCM, take the minimum value
for iexp=1:10
  [P,U,g,S,J,t,errMin] = spfcm(x,c,const,'gamma',gamma,'a',0.5,'b',b,'distance','Mahalanobis','coef',1);
  if Jmin>J
    Jmin=J;
    Pmin=P;
  end
end
P=Pmin;

clust=sign(P-repmat(max(P')',1,c))+1; % get hard partition
clust=clust(:,2)*2+clust(:,1);

figure();hold on; % visualization
idx=find(clust==1);plot(x(idx,1),x(idx,2),'kx','markersize',8);
idx=find(clust==2);plot(x(idx,1),x(idx,2),'ko','markersize',8);
title('SPFCM-mah hard partition');
axis equal tight



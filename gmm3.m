function varargout=gmm3(X,K,imgid)
id=imgid;%ͼƬ����
iter=0; %�鿴�������������ڵ���
threshold=1e-10; %��¼����
[N,D]=size(X);
Randk=randperm(N);%���ѡȡk�����ݵ�
centroids=X(Randk(1:K),:);%����K������
[pMiu,pPi,pSigma]=init_params();
Lprev=-inf;%��һ�ξ�������??
%%E step
while iter<500 %���͵���500��
iter=iter+1;
if any(isnan(pSigma))==0   %���û�з����쳣*
Px=calc_prob();%�������ɸ��ʾ���ĺ���
else
pMiu=[];  %�쳣�����ԭ����������ά�����������Э����������죬�����ϲ���һ�£��Ѿ������˽�������
pPi=[];   %������Ǵ���ʧ�ܵĻ�����������ͼ��������ͼ�����ݲ���
pSigma=[];
   break ;
end
pGamma=Px.*repmat(pPi,N,1);
pGamma=pGamma./repmat(sum(pGamma,2),1,K);%pGamma(i,k)����Xi�ɵ�k��Gaussian���ɵĸ���
%%M step
Nk=sum(pGamma,1);%Nk��ʶ��k����˹����ÿ�������ĸ��ʵĺͣ�����Nk���ܺ�ΪN
pMiu=diag(1./Nk)*pGamma'*X;%update?pMiu?through?MLE(ͨ�����?=?0�õ�)
pPi=Nk/N;
%update k��pSigma
for kk=1:K
Xshift=X-repmat(pMiu(kk,:),N,1);
pSigma(:,:,kk)=(Xshift'*...
(diag(pGamma(:,kk))*Xshift))/Nk(kk);
pSigma(:,:,kk)=pSigma(:,:,kk)+eye(27);  %����sigma�������������ĶԽ���Ԫ��
end
%���㾫��
L=sum(log(Px*pPi'));
if L-Lprev<threshold
break;
end
Lprev =L;
end
if nargout==1
model=[];
model.Miu=pMiu;
model.Sigma=pSigma;
model.Pi=pPi;
varargout={model};
else
model=[];
model.Miu=pMiu;
model.Sigma=pSigma;
model.Pi=pPi;
varargout={Px,model};
end

function [pMiu,pPi,pSigma]=init_params() %��ʼ�������Ӻ���
pMiu=centroids;
pPi=zeros(1,K);%k��GMM��ռȨ��
pSigma=zeros(D,D,K);%k��GMM��Э�������ÿ����D*D��
%������󣬼���N*K�ľ���x-pMiu��^2=x^2+pMiu^2-2*x*Miu
distmat=repmat(sum(X.*X,2),1,K)+...%x^2,?N*1�ľ���replicateK��
repmat(sum(pMiu.*pMiu,2)',N,1)-...%pMiu^2��1*K�ľ���replicateN��
2*X*pMiu';
[~,labels]=min(distmat,[],2);%label����Сֵ���ڵ��кţ������
for k=1:K
Xk=X(labels==k,:); %ѡ��X�б���ʼ�鵽��k�����������ݹ��ɾ���
pPi(k)=size(Xk,1)/N;  %����Ȩ����Xk�е���������/��ȡ����
pSigma(:,:,k)=cov(Xk)+eye(27);   %��Э����
end
end
function Px=calc_prob()
Px=zeros(N,K);  %Px��¼ÿ������ĳ�����������ĸ���
for k=1:K
Xshift=X-repmat(pMiu(k,:),N,1);%X-pMiu
inv_pSigma=inv(pSigma(:,:,k));
tmp=sum((Xshift*inv_pSigma).*Xshift,2);
coef=(2*pi)^(-D/2)*sqrt(det(inv_pSigma));
Px(:,k)=coef*exp(-0.5*tmp);
end
    end
end
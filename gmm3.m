function varargout=gmm3(X,K,imgid)
id=imgid;%图片名字
iter=0; %查看迭代次数，用于调试
threshold=1e-10; %记录精度
[N,D]=size(X);
Randk=randperm(N);%随机选取k个数据点
centroids=X(Randk(1:K),:);%构成K个中心
[pMiu,pPi,pSigma]=init_params();
Lprev=-inf;%上一次聚类的误差??
%%E step
while iter<500 %最大就迭代500次
iter=iter+1;
if any(isnan(pSigma))==0   %如果没有发生异常*
Px=calc_prob();%调用生成概率矩阵的函数
else
pMiu=[];  %异常情况，原因是由数据维数过高引起的协方差矩阵奇异，在网上查了一下，已经补充了矫正代码
pPi=[];   %如果还是处理失败的话就跳过这张图，用其他图的数据补充
pSigma=[];
   break ;
end
pGamma=Px.*repmat(pPi,N,1);
pGamma=pGamma./repmat(sum(pGamma,2),1,K);%pGamma(i,k)就是Xi由第k个Gaussian生成的概率
%%M step
Nk=sum(pGamma,1);%Nk标识第k个高斯生成每个样本的概率的和，所有Nk的总和为N
pMiu=diag(1./Nk)*pGamma'*X;%update?pMiu?through?MLE(通过令导数?=?0得到)
pPi=Nk/N;
%update k个pSigma
for kk=1:K
Xshift=X-repmat(pMiu(kk,:),N,1);
pSigma(:,:,kk)=(Xshift'*...
(diag(pGamma(:,kk))*Xshift))/Nk(kk);
pSigma(:,:,kk)=pSigma(:,:,kk)+eye(27);  %避免sigma矩阵非正定而填补的对角线元素
end
%计算精度
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

function [pMiu,pPi,pSigma]=init_params() %初始化参数子函数
pMiu=centroids;
pPi=zeros(1,K);%k类GMM所占权重
pSigma=zeros(D,D,K);%k类GMM的协方差矩阵，每个是D*D的
%距离矩阵，计算N*K的矩阵（x-pMiu）^2=x^2+pMiu^2-2*x*Miu
distmat=repmat(sum(X.*X,2),1,K)+...%x^2,?N*1的矩阵replicateK列
repmat(sum(pMiu.*pMiu,2)',N,1)-...%pMiu^2，1*K的矩阵replicateN行
2*X*pMiu';
[~,labels]=min(distmat,[],2);%label是最小值所在的列号，即类号
for k=1:K
Xk=X(labels==k,:); %选出X中被初始归到第k个分量的数据构成矩阵
pPi(k)=size(Xk,1)/N;  %它的权重是Xk中的样本数量/总取样点
pSigma(:,:,k)=cov(Xk)+eye(27);   %求协方差
end
end
function Px=calc_prob()
Px=zeros(N,K);  %Px记录每个点由某个分量产生的概率
for k=1:K
Xshift=X-repmat(pMiu(k,:),N,1);%X-pMiu
inv_pSigma=inv(pSigma(:,:,k));
tmp=sum((Xshift*inv_pSigma).*Xshift,2);
coef=(2*pi)^(-D/2)*sqrt(det(inv_pSigma));
Px(:,k)=coef*exp(-0.5*tmp);
end
    end
end
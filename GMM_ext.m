function varargout=GMM_ext(models,K)
%按照书上81页的公式翻译成代码，扩展EM算法  by 彭小雨2016140137
[~,ImgNum]=size(models);  %图片数量
[~,perImg]=size(models(1).Pi); %每幅图包含的高斯分量数目
paicm=zeros(1,K);   %设置新的GMM，将训练集产生的 ImgNum*perImg 个高斯模型拟合为K个
miucm=zeros(27,K);   %存放新的拟合参数  在processImg.m文件中将特征向量维数从192缩减到27维
sigmacm=zeros(27,27,K);
hjkm=zeros(ImgNum,perImg,K);
wjkm=zeros(ImgNum,perImg,K);
accr_prev=-inf;%上一次聚类的误差
threshold=1e-10; %记录精度
for i=1:perImg:K
    if(i+perImg-1>K) %避免越界访问
    paicm(1,i:K)=models(i).Pi(1,1:K-i+1);  %选取一些图片的GMM作为初值
    miucm(:,i:K)=models(i).Miu(:,1:K-i+1)';
    sigmacm(:,:,i:K)=models(i).Sigma(:,:,1:K-i+1);
        break; 
    end 
    paicm(1,i:i+perImg-1)=models(i).Pi;  %选取一些图片的GMM作为初值
    miucm(:,i:i+perImg-1)=models(i).Miu';
    sigmacm(:,:,i:i+perImg-1)=models(i).Sigma;
end
    
    iter=0;%记录迭代次数  
    
    while iter<500  %最大迭代次数
        %%E step
        for j=1:1:ImgNum
            for k=1:1:perImg
                miujk=models(j).Miu(k,:)'; % 保存当前要用来计算的高斯分量
                    sigmajk=models(j).Sigma(:,:,k);
                    paijk=models(j).Pi(1,k);
                    sum_hjkm=0;
                for m=1:1:K      %先把每个高斯分量由第m个高斯分量产生的概率算出来然后求和，为书上式3.10的分母  
        hjkm(j,k,m)=(Gauss(miujk,miucm(:,m),sigmacm(:,:,m))*exp((-1/2)*trace(inv(sigmacm(:,:,m))*sigmajk)))^paijk*paicm(1,m);
               sum_hjkm=sum_hjkm+hjkm(j,k,m);    
                end
                hjkm(j,k,:)=hjkm(j,k,:)/sum_hjkm ;
            end
        end
       %%M step 
       for m=1:1:K
           a=hjkm(:,:,m);   %平均有多少个分量是由第m个分量产生的
           paicm(1,m)=sum(a(:))/(ImgNum*perImg);%更新权重
       end
       for m=1:1:K
           sum_wjkm=0;
           for j=1:1:ImgNum
               for k=1:1:perImg
        % 更新权重wjkm 需要用到hjkm和每个要聚合的分量的权重
                  
                    paijk=models(j).Pi(1,k);
                   wjkm(j,k,m)=hjkm(j,k,m)*paijk;
                   sum_wjkm=sum_wjkm+wjkm(j,k,m);
               end
           end
           wjkm(:,:,m)=wjkm(:,:,m)/sum_wjkm ;
           miucm_m_new=zeros(27,1);%更新均值和协方差矩阵
           sigmacm_m_new=zeros(27,27);
            for j=1:1:ImgNum       %更新均值miucm
               for k=1:1:perImg
                    miujk=models(j).Miu(k,:)'; % 读取每个分量的均值
                    miucm_m_new=miucm_m_new+wjkm(j,k,m)*miujk;
               end
            end
           miucm(:,m)=miucm_m_new;
           
            for j=1:1:ImgNum  %更新协方差矩阵sigmacm
               for k=1:1:perImg
                    miujk=models(j).Miu(k,:)'; % 读取每个分量的均值
                    sigmajk=models(j).Sigma(:,:,k);%读取每个分量的协方差矩阵
                    sigmacm_m_new=sigmacm_m_new+wjkm(j,k,m)*(sigmajk+(miujk-miucm_m_new)*(miujk-miucm_m_new)');
               end
            end
            sigmacm(:,:,m)=sigmacm_m_new;         
       end
       accr=0;
       for m=1:1:K   %计算精度
       a=hjkm(:,:,m);  
       accr=accr+log(sum(a(:))*paicm(1,m));%每一个分量产生各个点的概率加权取对数再求和
       end
       if accr-accr_prev<threshold
           break;
       end
       accr_prev=accr;
    end
model=[];                  %迭代结束，封装结果
model.Miu=miucm;
model.Sigma=sigmacm;
model.Pi=paicm;
varargout={model};
 end
 function y = Gauss(X,Mu,Conv) %多维高斯函数的概率密度函数
Conv = Conv+eye(27);
[n,~] = size(X);
y = 1/((2*pi)^(n/2)*sqrt(det(Conv)))*exp(-0.5*(X-Mu)'/Conv*(X-Mu));
end
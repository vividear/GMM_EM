function varargout=GMM_ext(models,K)
%��������81ҳ�Ĺ�ʽ����ɴ��룬��չEM�㷨  by ��С��2016140137
[~,ImgNum]=size(models);  %ͼƬ����
[~,perImg]=size(models(1).Pi); %ÿ��ͼ�����ĸ�˹������Ŀ
paicm=zeros(1,K);   %�����µ�GMM����ѵ���������� ImgNum*perImg ����˹ģ�����ΪK��
miucm=zeros(27,K);   %����µ���ϲ���  ��processImg.m�ļ��н���������ά����192������27ά
sigmacm=zeros(27,27,K);
hjkm=zeros(ImgNum,perImg,K);
wjkm=zeros(ImgNum,perImg,K);
accr_prev=-inf;%��һ�ξ�������
threshold=1e-10; %��¼����
for i=1:perImg:K
    if(i+perImg-1>K) %����Խ�����
    paicm(1,i:K)=models(i).Pi(1,1:K-i+1);  %ѡȡһЩͼƬ��GMM��Ϊ��ֵ
    miucm(:,i:K)=models(i).Miu(:,1:K-i+1)';
    sigmacm(:,:,i:K)=models(i).Sigma(:,:,1:K-i+1);
        break; 
    end 
    paicm(1,i:i+perImg-1)=models(i).Pi;  %ѡȡһЩͼƬ��GMM��Ϊ��ֵ
    miucm(:,i:i+perImg-1)=models(i).Miu';
    sigmacm(:,:,i:i+perImg-1)=models(i).Sigma;
end
    
    iter=0;%��¼��������  
    
    while iter<500  %����������
        %%E step
        for j=1:1:ImgNum
            for k=1:1:perImg
                miujk=models(j).Miu(k,:)'; % ���浱ǰҪ��������ĸ�˹����
                    sigmajk=models(j).Sigma(:,:,k);
                    paijk=models(j).Pi(1,k);
                    sum_hjkm=0;
                for m=1:1:K      %�Ȱ�ÿ����˹�����ɵ�m����˹���������ĸ��������Ȼ����ͣ�Ϊ����ʽ3.10�ķ�ĸ  
        hjkm(j,k,m)=(Gauss(miujk,miucm(:,m),sigmacm(:,:,m))*exp((-1/2)*trace(inv(sigmacm(:,:,m))*sigmajk)))^paijk*paicm(1,m);
               sum_hjkm=sum_hjkm+hjkm(j,k,m);    
                end
                hjkm(j,k,:)=hjkm(j,k,:)/sum_hjkm ;
            end
        end
       %%M step 
       for m=1:1:K
           a=hjkm(:,:,m);   %ƽ���ж��ٸ��������ɵ�m������������
           paicm(1,m)=sum(a(:))/(ImgNum*perImg);%����Ȩ��
       end
       for m=1:1:K
           sum_wjkm=0;
           for j=1:1:ImgNum
               for k=1:1:perImg
        % ����Ȩ��wjkm ��Ҫ�õ�hjkm��ÿ��Ҫ�ۺϵķ�����Ȩ��
                  
                    paijk=models(j).Pi(1,k);
                   wjkm(j,k,m)=hjkm(j,k,m)*paijk;
                   sum_wjkm=sum_wjkm+wjkm(j,k,m);
               end
           end
           wjkm(:,:,m)=wjkm(:,:,m)/sum_wjkm ;
           miucm_m_new=zeros(27,1);%���¾�ֵ��Э�������
           sigmacm_m_new=zeros(27,27);
            for j=1:1:ImgNum       %���¾�ֵmiucm
               for k=1:1:perImg
                    miujk=models(j).Miu(k,:)'; % ��ȡÿ�������ľ�ֵ
                    miucm_m_new=miucm_m_new+wjkm(j,k,m)*miujk;
               end
            end
           miucm(:,m)=miucm_m_new;
           
            for j=1:1:ImgNum  %����Э�������sigmacm
               for k=1:1:perImg
                    miujk=models(j).Miu(k,:)'; % ��ȡÿ�������ľ�ֵ
                    sigmajk=models(j).Sigma(:,:,k);%��ȡÿ��������Э�������
                    sigmacm_m_new=sigmacm_m_new+wjkm(j,k,m)*(sigmajk+(miujk-miucm_m_new)*(miujk-miucm_m_new)');
               end
            end
            sigmacm(:,:,m)=sigmacm_m_new;         
       end
       accr=0;
       for m=1:1:K   %���㾫��
       a=hjkm(:,:,m);  
       accr=accr+log(sum(a(:))*paicm(1,m));%ÿһ����������������ĸ��ʼ�Ȩȡ���������
       end
       if accr-accr_prev<threshold
           break;
       end
       accr_prev=accr;
    end
model=[];                  %������������װ���
model.Miu=miucm;
model.Sigma=sigmacm;
model.Pi=paicm;
varargout={model};
 end
 function y = Gauss(X,Mu,Conv) %��ά��˹�����ĸ����ܶȺ���
Conv = Conv+eye(27);
[n,~] = size(X);
y = 1/((2*pi)^(n/2)*sqrt(det(Conv)))*exp(-0.5*(X-Mu)'/Conv*(X-Mu));
end
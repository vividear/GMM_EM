function GMM_model=main()
for i=1:1:100  
x=['183000/',num2str(183000+i-1),'.jpeg'];  %选取183000图像集合
A=imread(x);
X=processImg(A);
models(i)=gmm3(X',8,x);
if isempty(models(i).Pi) 
models(i)=models(i-1);
end
end
GMM_model=GMM_ext(models,64);
end
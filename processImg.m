function [ result ] = processImg( X )
%UNTITLED3 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
A=X;
Ar=A(:,:,1);%�ֳ�RGB����ͨ��
Ag=A(:,:,2);
Ab=A(:,:,3);
result=zeros(27,5673);  %��5673�������㣬ÿ��������ȡ27ά
[M,N]=size(Ar);
k=1;
for i=1:2:(M-7)   %ÿ��ͼ��8*8���أ������ش��ڻ���
  for j=1:2:(N-7)
     ablkr = Ar(i:i+7,j:j+7);
     ablkg = Ag(i:i+7,j:j+7);
     ablkb = Ab(i:i+7,j:j+7);
     b = dct2(ablkr);
     d=[b(1,1);b(1,2);b(1,3);b(2,1);b(2,2);b(2,3);b(3,1);b(3,2);b(3,3)];
     b = dct2(ablkg);
     d=[d;b(1,1);b(1,2);b(1,3);b(2,1);b(2,2);b(2,3);b(3,1);b(3,2);b(3,3)];
     b = dct2(ablkb);
     d =[d;b(1,1);b(1,2);b(1,3);b(2,1);b(2,2);b(2,3);b(3,1);b(3,2);b(3,3)];
     result(:,k)=d;
     k=k+1;
  end
end


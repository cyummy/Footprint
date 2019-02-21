clear;
close all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%��һ����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%������ͼ�����һϵ��Ԥ���� ������ȡ������
%1�����û���任���ֱ�� ��ͼ���г��ӳ���������ˮƽ
%2��ȥ��ԭͼ��ԭ�еİױ߻���������ת�����в����ĺڱ� ��������Գ��ӵļ���������
%3������һ�����������û���任������λ�ã�����������ȡ����
%4���ٴ���������λ����ˮƽ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ͨ������ ��÷���ˮƽ�ĳ���

for c=1:1:6  %�������δ�������㼣ͼ�� �����ɸ��ݾ��崦����Ҫ���е���
Img = imread(['C:\Users\pc\Desktop\�㼣\�㼣ͼ��\footprint (',num2str(c),').jpg']);%��ȡ�㼣ͼ���·��
Img_gary = rgb2gray(Img);%�ҶȻ�
%subplot(1,3,1),imshow(Img_gary),title('Img gary');
Img_edge = edge(Img_gary,'sobel');  %���ؼ��
%subplot(1,3,2),imshow(Img_edge),title('Img edge');
se = strel('line',5,0); %�ṹԪ�� ���� �Ƕ�Ϊ0  ���ȸ����������
Img_erode = imerode(Img_edge,se);%��ʴ��ɫǰ������ ��������
%subplot(1,3,3),imshow(Img_erode),title('Img erode');

[H,T,R] = hough(Img_erode);%����任 ���ֱ�� ������
P = houghpeaks(H,1,'threshold',ceil(0.5*max(H(:))));
x1 = T(P(:,2));  %theta 
y1 = R(P(:,1));  %rho

if (x1 > 0)     %ˮƽ����theta��90�� ��ֱ����ת��ˮƽ����
   Img_rotate = imrotate(Img_gary, abs(x1)-90,'bilinear','crop');
else
   Img_rotate = imrotate(Img_gary, 90-abs(x1),'bilinear','crop');
end
%figure,imshow(Img_rotate),title('Img_rotate');

%%%%%%%%%%%%%%%%%%%%%��ԭͼ�����ϼ��У�ȥ����ת�����в����ĺڱ߻�ͼ��ԭ�еİױ�
[m,n] = size(Img_rotate);     %m��ͼ�����򳤶� n�Ǻ��򳤶�
Img_max = 0;
Img_min = m;
for i = ceil(0.2*n):10:ceil(0.8*n)
    for j = 1:10:m
         if  Img_rotate(j,i)> 0 && Img_rotate(j,i)< 255
            break;
         end
    end
    if j > Img_max
        Img_max = j;
    end
   
    for l = m:-10:1
        if Img_rotate(l,i)> 0 && Img_rotate(l,i)< 255
            break;
        end
    end
    if l < Img_min
        Img_min = l;
    end
end
Img_height = Img_min-Img_max;

[Img_crop,rect] = imcrop(Img_rotate,[ceil(0.2*n),Img_max,ceil(0.6*n),Img_height]);   %���������꣬��͸�
%figure,imshow(Img_crop),title('Img crop');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%��������任���ͼ���е�ֱ�� ��������ȡ����
Img_edge1 = edge(Img_crop,'sobel','horizontal');    %ˮƽ�����ϱ��ؼ��
se = strel('line',6,0);
Img_erode1 = imerode(Img_edge1,se);%ͼ��ʴ
%figure,imshow(Img_erode1),title('��ת���Եͼ��');

[H,T,R] = hough(Img_erode1,'RhoResolution',1,'Theta',-90:0.1:-88);  %��ͶƱ�ռ�theta��Χ ��С��88-90
%figure;
%imshow(H,[],'XData',T,'YData',R,'InitialMagnification','fit');
%xlabel('\theta'),ylabel('\rho');
%axis on,axis normal,hold on;

P = houghpeaks(H,6,'threshold',ceil(0.3*max(H(:))));
x = T(P(:,2));
y = R(P(:,1));
%plot(x,y,'s','color','white')

lines = houghlines(Img_erode1,T,R,P,'FillGap',650,'MinLength',90);  
figure,imshow(Img_crop);
hold on;
max_len = 0;
for k = 1:length(lines)
    xy = [lines(k).point1;lines(k).point2];
%    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
%��עֱ�߶εĶ˵�    
%plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
h(k) = (xy(1,2)+xy(2,2))/2;
end

 %��СROI����Χ ��������ȡ����
 h_max = max(h(:));
 h_min = min(h(:)); 
 height = h_max-h_min;
 if height <  20 
    Img_ruler = imcrop(Img_crop,[ceil(0.2*n),h_min-10,ceil(0.6*n),h_max-h_min+30]);    
 else
    Img_ruler = imcrop(Img_crop,[ceil(0.2*n),h_min,ceil(0.6*n),(h_max-h_min)]); 
 end
 %figure,imshow(Img_ruler),title('Img ruler');
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%�ٴ���������λ����ˮƽ
Img_edge2 = edge(Img_ruler,'sobel');    %���ؼ��
se = strel('line',5,0); %�ṹԪ�� ���� �Ƕ�Ϊ0  ���ȸ����������
Img_erode2 = imerode(Img_edge2,se);%ͼ��ʴ
%figure,imshow(Img_erode2);
[H,T,R] = hough(Img_erode2);
P = houghpeaks(H,2,'threshold',ceil(0.5*max(H(:))));              %5
x2 = mean(T(P(:,2)));  %theta

if (x2 > 0)     %ˮƽ����theta��90��
   Img_rotate2 = imrotate(Img_ruler, abs(x2)-90,'bilinear','crop');
else
   Img_rotate2 = imrotate(Img_ruler, 90-abs(x2),'bilinear','crop');
end
%figure,imshow(Img_rotate2),title('Img ruler1');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%�ڶ�����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�Գ��Ӷ�ֵ�� �����ݺ�ɫ�����ڶ�ֵͼ������ռ�����Գ������������ж� ��Ϊ�ڳߺͰ׳�
%�������� �Ժڳߡ��׳߲�ȡ��ͬ�Ĵ������ 
%   �ڳ�
%1����ROI��С���̶����������� �ӳ��ӵĶ�ֵͼ������ȡ�̶�������
%   �б�����ֵͼ�� ���ĳ�а�ɫ������ռ������ĳһ��Χ�� ����Ϊ�����ǿ̶� �õ��̶ȵ�λ����Ϣ ������Ӧͼ���н��п̶�����ȡ
%2���Կ̶��߽���ȥë�̺Ͳ�ȱ���� ��ߺ�����⾫��
%3��ͨ���ڶ�ֵͼ����Ѱ�Һڰ׽��紦 ȷ��ÿ���̶��ߵ�λ�� ������С���˷���� ����ͼ�п̶ȳ�1cm�����������������ظ��� a0
%
%   �׳�
%1��������ֵ���¶�ֵ��
%2����ROI��С���̶����������� �ӳ��ӵĻҶ�ͼ������ȡ���̶�������
%   �б�����ֵͼ�� ���ĳ�а�ɫ������ռ������ĳһ��Χ�� ����Ϊ�����ǿ̶� �õ��̶ȵ�λ����Ϣ ������Ӧͼ���н��п̶�����ȡ
%3��ͨ���ڻҶ�ͼ����Ѱ�Ҿֲ���Сֵ ȷ��ÿ���̶��ߵ�λ�� ������С���˷���� ����ͼ�п̶ȳ�1cm�����������������ظ��� a0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Otsu��ֵ��������
 [level EM] = graythresh(Img_rotate2);
 Img_BW1 =im2bw(Img_rotate2,level);
 %figure,imshow(Img_BW1),title('Img bw');

[m,n] = size(Img_BW1); 
 k=0;
for i =1:n
     for j = 1:m
         if Img_BW1(j,i) == 0
             k=k+1;
         end
     end
end
P=k/(m*n);  %��ɫ�����ڶ�ֵͼ������ռ����

%%%%%%%%%%%%%%%%%%%%�ڳ�%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if P > 0.2       
    M=zeros(m+1);
    V=zeros(m+1);
    ROI_height=0;
    ROI_max=0;
    F=1;
    count=0;
    Thresh = 5;   %�̶���������ֵ   �ɵ���  
    for i=1:m
        for j=1:n
            if Img_BW1(i,j)==1
                V(i) = V(i)+1;
            end
        end
        M(i) = V(i)/n;
    end  

   %��ROI��С���̶�����������
    for i=1:1:m
        if  M(i)>0.25 && M(i)<0.65 && F==1    %M��i��������а�ɫ���ص���ռ���� �����ĳ��Χ�ڣ�������Ϊ�����ǿ̶�
            count = count + 1;
        else
            count = 0;
        end
    
        if (count > Thresh) && ((M(i+1)<0.25) || (M(i+1)>0.65)) && M(i)>0.25 && M(i)<0.65 
           F=0;      %��־λ��0 ��һ���ҵ�����Ҫ��Ŀ̶����������ֹͣ��Ѱ
           ROI_max = i ;
           ROI_height = count;
        end
    end 
    Img_ROI = imcrop(Img_BW1,[ceil(0.2*n), ROI_max-ROI_height,ceil(0.6*n),ROI_height]); 
    %figure,imshow(Img_ROI),title('Img ROI');

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %Ϊ����ߺ�����⾫��  �Կ̶��߽���ȥë�̺Ͳ�ȱ���� 
  %ȥë�� ��ȱ�� �ṹԪ�ط�
  %ȥë�̣�����ÿ�� �����������ɫ�����Ҹ���С��N ����Ϊ��ë�� ����Ҷ�ֵ��Ϊ��ɫ ��ȥ��ë��
  %
  %��ȥë�̻����� ��ȱ�� ������ȥë������
  %��ȱ������ÿ�� �����������ɫ�����Ҹ���С��N ����Ϊ��ȱ�𣨿׶��� ����Ҷ�ֵ��Ϊ��ɫ ����ȱ��
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %ȥë��
  N=5;
  [m,n] = size(Img_ROI);
  K = zeros(m,n);
  for i = 1:n
      h=1;
      for j = 1:m-1
          if Img_ROI(j,i) ==1
             K(h,i) = K(h,i) + 1;
              if (Img_ROI(j,i)==1) & (Img_ROI(j+1,i)==0) & (K(h,i)< N)
                  for t=0:(K(h,i)-1)
                      Img_ROI(j-t,i) = 0; 
                  end
                  h = h+1;
              end
          end
      end
  end
  %figure,imshow(Img_ROI),title('Img ROI1');

 %��ȱ��
  N=ceil(ROI_height/2); 
  [m,n] = size(Img_ROI);
  K = zeros(m,n);
  for i = 1:n
      h=1;
      for j = 1:m-1
          if Img_ROI(j,i) ==0
              K(h,i) = K(h,i) + 1;
              if (Img_ROI(j,i)==0) & (Img_ROI(j+1,i)==1) & (K(h,i)< N)
                  for t=0:(K(h,i)-1)      
                      Img_ROI(j-t,i) = 1; 
                  end
                  h = h+1;
              end
          end
      end
  end
 % figure,imshow(Img_ROI),title('Img ROI2');

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %������С���˷� ����ͼ�п̶ȳ�1cm�����������������ظ��� a0
  %���������������� �޳��쳣���� ��������������������С��� ��߾���
  %1���������������� Ҫ������������ڽϿ�Ŀ̶ȳ� ��ȡ��2d+1�������ݣ��Խ�խ�Ŀ̶ȳ� ֻȡ���м�������
  %2���޳��쳣���ݣ� ��ԭʼ����������ͨ����С���˷��������Ƴ�һ���̶��߼���������������� �Դ�Ϊ�ο��޳��쳣���� 
  %3��������������� ����10���̶ȣ���1cm����ڵ����ظ�������1mm��ߵ�1cm
  %
  % H������ÿ���̶��ߵ�λ������
  % G�����Ÿ��̶��߼���ڵ����ص���
  % R�����ڴ���޳��쳣���ݺ�̶��߼���ڵ����ظ��� ��mmΪ��λ
  % S�����ڴ���޳��쳣���ݺ�10���̶��߼���ڵ����ظ��� ��cmΪ��λ
  % T�����ڴ��S����������ת���õ���λ����Ϣ ������С���˷�����
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   [m,n] = size(Img_ROI);  
   d=3; 
   %��С����1 ��ȡ���� ������������
   if ceil(ROI_height/2)>d   
      H = zeros(2*d+1,n);     %H������ÿ���̶��ߵ�λ��
      G = zeros();            %G�����Ÿ��̶��߼���ڵ����ص���
      E = zeros(); 
      h=0;
      f=0;
      for i=(ceil(ROI_height/2)-d):(ceil(ROI_height/2)+d)
          h=h+1;%����
          H(h,1)=0;
          k=1;
          for j = 1:n
              if Img_ROI(i,j)==1 && Img_ROI(i,j-1)==0
                 k=k+1;
                 H(h,k)=j;
                 f=f+1;
                 G(f)=H(h,k)-H(h,k-1);
              end
          end
          E(h) = k;%E�����Ÿ��п̶��߸���
      end
      
 %ͨ����С���˷��������Ƴ�һ���̶��߼���������������� �Դ�Ϊ�ο��޳��쳣����      
 x=[1:1:E(d+1)-1];
 y=H(d+1,2:1:E(d+1));
 p=polyfit(x,y,1);%��С�������
 a1=polyval(p,1);
 a2=polyval(p,2);
 a=a2-a1;
 
 %��С����2 �޳��쳣����
 R = zeros();   %R�����ڴ���޳��쳣���ݺ�̶��߼���ڵ����ظ��� ��mmΪ��λ
 r=0;
 N=3;
 for i =1:f
     if abs(ceil(G(i)-a))<N
         r=r+1;
         R(r)=G(i);
     else
         continue
     end
 end
 
   %%%%ֻȡ���м������ݽ��м���     
   else
     i = ceil(ROI_height/2); 
     H = zeros(n);
     G = zeros(n);
     k=0;
     for j = 1:n
        if Img_ROI(i,j)==1 && Img_ROI(i,j-1)==0
           k=k+1;
           H(k)=j;
        end
     end
 
     for j=1:k-1
         G(j)=H(j+1)-H(j);
     end

 x=[1:1:k];
 y=H(1:1:k);
 p=polyfit(x,y,1);  %��С���˷����
 a1=polyval(p,1);
 a2=polyval(p,2);
 a=a2-a1;
    
  %�޳��쳣����
  R = zeros(); 
  r=0;
  N=3;
     for i =1:k-1
         if abs(ceil(G(i)-a))<N
           r=r+1;
           R(r)=G(i);
         else
             continue
         end
     end  
     
   end
%%%%%%%%%��С����3 ����������� ����10���̶ȣ���1cm����ڵ����ظ���
  S = zeros(n);    %S�����ڴ���޳��쳣���ݺ�10���̶��߼���ڵ����ظ��� ��cmΪ��λ
  s=1;
  for i=1:r-9
      for j=0:9
          S(s)=S(s)+R(i+j);
      end
      s=s+1;
  end
   
T = zeros(n); %ת��Ϊλ����Ϣ ������С���˷�����
t=1;
T(1)=0;
for i=1:s 
    t=t+1;
    T(t)=T(t-1)+S(i);
end
    
x=[1:1:t];
y=T(1:1:t);
p=polyfit(x,y,1);
a1=polyval(p,1);
a2=polyval(p,2);
a0=a2-a1;          
%plot(x,y,'*');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%�׳�%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else
 level=0.8; %������ֵ���¶�ֵ��
 Img_BW2 =im2bw(Img_rotate2,level);
 %figure,imshow(Img_BW2);
    
    %��ȡ�̶�������
    M=zeros(m);
    N=zeros(m);
    ROI_height=0;
    ROI_max=0;
    F=1;
    count=0;
    Thresh =3;   %�ɵ���
    for i=1:m
        for j=1:n
            if Img_BW2(i,j)==1
                N(i) = N(i)+1;
            end
        end
        M(i) = N(i)/n;
    end
    
    for i=1:m
        if  M(i)>0.3 && M(i)<0.7 && F==1    %M��i��������а�ɫ���ص���ռ���� �����ĳ��Χ�ڣ�������Ϊ�ǿ̶�������
            count = count + 1;
        else
            count = 0;
        end
    
        if (count > Thresh) && ((M(i+1)<0.3) || (M(i+1)>0.7)) && M(i)>0.3 && M(i)<0.7
           F=0;
           ROI_max = i ;
           ROI_height = count;
        end
    end 
    
    Img_ROI = imcrop(Img_rotate2,[ceil(0.2*n), ROI_max-ROI_height,ceil(0.6*n),ROI_height]); 
  %  figure,imshow(Img_ROI);   %ROIΪ�Ҷ�ͼ��

   [m,n] = size(Img_ROI);  
    d=1; 
   
   %��С����ֶ�1 ��ȡ���� ������С���˷��������ݻ���
   if ceil(ROI_height/2)>d   
      H = zeros(2*d+1,n);     %ÿ���̶��ߵ�λ��
      G = zeros();            %���̶���֮������ص���
      E = zeros();
      h=0;
      f=0;
      for i=(ceil(ROI_height/2)-d):(ceil(ROI_height/2)+d)
          h=h+1;
          H(h,1)=0;
          k=1;
          for j = 2:n-1
              if Img_ROI(i,j)<Img_ROI(i,j-1) && Img_ROI(i,j)<=Img_ROI(i,j+1)
                 k=k+1;
                 H(h,k)=j;
                 f=f+1;
                 G(f)=H(h,k)-H(h,k-1);
              end
          end
          E(h) = k;
      end

 %ͨ����С���˷��������Ƴ�һ���̶��߼���������������� �Դ�Ϊ�ο��޳��쳣����    
 x=[1:1:E(d+1)-1];
 y=H(d+1,2:1:E(d+1));
 p=polyfit(x,y,1);
 a1=polyval(p,1);
 a2=polyval(p,2);
 a=a2-a1;
 
 %��С����ֶ�2 �޳��쳣����
 R = zeros();   %�޳��쳣���ݺ�̶��߼����ظ��� ��mmΪ��λ
 r=0;
 N=3;
 for i =1:f
     if abs(ceil(G(i)-a))<N
         r=r+1;
         R(r)=G(i);
     else
         continue
     end
 end
      
   else
     %%%%ֻȡ���м������ݽ��м���  
     i = ceil(ROI_height/2); 
     H = zeros(n);
     G = zeros(n);
     k=0;
     for j = 2:n-1
         if Img_ROI(i,j)<Img_ROI(i,j-1) && Img_ROI(i,j)<=Img_ROI(i,j+1)
            k=k+1;
            H(k)=j;
         end
      end
 
      for j=1:k-1
          G(j)=H(j+1)-H(j);
      end

  %ͨ����С���˷��������Ƴ�һ���̶��߼���������������� �Դ�Ϊ�ο��޳��쳣����  
  x=[1:1:k];
  y=H(1:1:k);
  p=polyfit(x,y,1);
  a1=polyval(p,1);
  a2=polyval(p,2);
  a=a2-a1;
  
   %��С����ֶ�2 �޳��쳣����
   R = zeros(); 
   r=0;
   N=3;
      for i =1:k-1
          if abs(ceil(G(i)-a))<N
            r=r+1;
            R(r)=G(i);
          else
              continue
          end
      end  
      
   end
%%%%%%%%%��С����ֶ�3 ����10���̶ȣ���1cm֮������ظ���
 S = zeros(n);    %�޳��쳣���ݺ�10���̶��߼����ظ��� ��cmΪ��λ
  s=1;
  for i=1:r-9
      for j=0:9
          S(s)=S(s)+R(i+j);
      end
      s=s+1;
  end
   
T = zeros(n); %ת��Ϊλ����Ϣ ������С���˷�����
t=1;
T(1)=0;
for i=1:s 
    t=t+1;
    T(t)=T(t-1)+S(i);
end

%��С�������   
x=[1:1:t];
y=T(1:1:t);
p=polyfit(x,y,1);
a1=polyval(p,1);
a2=polyval(p,2);
a0=a2-a1;          
%plot(x,y,'*');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%��������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�ڶ����ֵõ�ͼ�г���10���̶��߼���� ��1cm������������
%ʵ�ʾ��������صĻ����ϵΪ 72����/1Ӣ�� ��28.3464578����/cm
%ȷ�������� ��ͼ�����ؾ�����ʵ�����ؾ���֮��ı�����ϵ
%���ñ����߽�ͼ�����ŵ�ʵ�ʴ�С
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%���ű�����
scale=28.3464578/a0;  
real_Img=imresize(Img,scale,'bilinear');
%figure,imshow(real_Img);
imwrite(real_Img,['C:\Users\pc\Desktop\�㼣\ʵ�ʴ�С�㼣ͼ��\real (',num2str(c),').bmp'],'bmp');%�洢ʵ�ʴ�С�㼣ͼ���·��


end

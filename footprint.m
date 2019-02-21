clear;
close all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%第一部分%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%对输入图像进行一系列预处理 从中提取出尺子
%1、利用霍夫变换检测直线 将图像中尺子初步修正至水平
%2、去除原图像原有的白边或在修正旋转过程中产生的黑边 避免后续对尺子的检测产生干扰
%3、在上一步基础上利用霍夫变换检测尺子位置，并将尺子提取出来
%4、再次修正尺子位置至水平
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%通过修正 获得方向水平的尺子

for c=1:1:6  %批量依次处理多张足迹图像 参数可根据具体处理需要自行调整
Img = imread(['C:\Users\pc\Desktop\足迹\足迹图像\footprint (',num2str(c),').jpg']);%读取足迹图像的路径
Img_gary = rgb2gray(Img);%灰度化
%subplot(1,3,1),imshow(Img_gary),title('Img gary');
Img_edge = edge(Img_gary,'sobel');  %边沿检测
%subplot(1,3,2),imshow(Img_edge),title('Img edge');
se = strel('line',5,0); %结构元素 线形 角度为0  长度根据整体调整
Img_erode = imerode(Img_edge,se);%腐蚀白色前景像素 减少噪声
%subplot(1,3,3),imshow(Img_erode),title('Img erode');

[H,T,R] = hough(Img_erode);%霍夫变换 检测直线 即尺子
P = houghpeaks(H,1,'threshold',ceil(0.5*max(H(:))));
x1 = T(P(:,2));  %theta 
y1 = R(P(:,1));  %rho

if (x1 > 0)     %水平方向theta是90度 将直线旋转至水平方向
   Img_rotate = imrotate(Img_gary, abs(x1)-90,'bilinear','crop');
else
   Img_rotate = imrotate(Img_gary, 90-abs(x1),'bilinear','crop');
end
%figure,imshow(Img_rotate),title('Img_rotate');

%%%%%%%%%%%%%%%%%%%%%在原图基础上剪切，去除旋转过程中产生的黑边或图像原有的白边
[m,n] = size(Img_rotate);     %m是图像纵向长度 n是横向长度
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

[Img_crop,rect] = imcrop(Img_rotate,[ceil(0.2*n),Img_max,ceil(0.6*n),Img_height]);   %起点横纵坐标，宽和高
%figure,imshow(Img_crop),title('Img crop');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%借助霍夫变换检测图像中的直线 将尺子提取出来
Img_edge1 = edge(Img_crop,'sobel','horizontal');    %水平方向上边沿检测
se = strel('line',6,0);
Img_erode1 = imerode(Img_edge1,se);%图像腐蚀
%figure,imshow(Img_erode1),title('旋转后边缘图像');

[H,T,R] = hough(Img_erode1,'RhoResolution',1,'Theta',-90:0.1:-88);  %将投票空间theta范围 缩小到88-90
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
%标注直线段的端点    
%plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
h(k) = (xy(1,2)+xy(2,2))/2;
end

 %缩小ROI区域范围 将尺子提取出来
 h_max = max(h(:));
 h_min = min(h(:)); 
 height = h_max-h_min;
 if height <  20 
    Img_ruler = imcrop(Img_crop,[ceil(0.2*n),h_min-10,ceil(0.6*n),h_max-h_min+30]);    
 else
    Img_ruler = imcrop(Img_crop,[ceil(0.2*n),h_min,ceil(0.6*n),(h_max-h_min)]); 
 end
 %figure,imshow(Img_ruler),title('Img ruler');
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%再次修正尺子位置至水平
Img_edge2 = edge(Img_ruler,'sobel');    %边沿检测
se = strel('line',5,0); %结构元素 线形 角度为0  长度根据整体调整
Img_erode2 = imerode(Img_edge2,se);%图像腐蚀
%figure,imshow(Img_erode2);
[H,T,R] = hough(Img_erode2);
P = houghpeaks(H,2,'threshold',ceil(0.5*max(H(:))));              %5
x2 = mean(T(P(:,2)));  %theta

if (x2 > 0)     %水平方向theta是90度
   Img_rotate2 = imrotate(Img_ruler, abs(x2)-90,'bilinear','crop');
else
   Img_rotate2 = imrotate(Img_ruler, 90-abs(x2),'bilinear','crop');
end
%figure,imshow(Img_rotate2),title('Img ruler1');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%第二部分%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%对尺子二值化 并依据黑色像素在二值图像中所占比例对尺子性质做出判断 分为黑尺和白尺
%分类讨论 对黑尺、白尺采取不同的处理策略 
%   黑尺
%1、将ROI缩小至刻度线所在区域 从尺子的二值图像中提取刻度线区域：
%   行遍历二值图像 如果某行白色像素所占比例在某一范围内 就认为该行是刻度 得到刻度的位置信息 并在相应图像中进行刻度线提取
%2、对刻度线进行去毛刺和补缺损处理 提高后续检测精度
%3、通过在二值图像上寻找黑白交界处 确定每个刻度线的位置 并用最小二乘法拟合 计算图中刻度尺1cm距离内所包含的像素个数 a0
%
%   白尺
%1、调高阈值重新二值化
%2、将ROI缩小至刻度线所在区域 从尺子的灰度图像中提取出刻度线区域：
%   行遍历二值图像 如果某行白色像素所占比例在某一范围内 就认为该行是刻度 得到刻度的位置信息 并在相应图像中进行刻度线提取
%3、通过在灰度图像上寻找局部最小值 确定每个刻度线的位置 并用最小二乘法拟合 计算图中刻度尺1cm距离内所包含的像素个数 a0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Otsu二值化方法：
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
P=k/(m*n);  %黑色像素在二值图像中所占比例

%%%%%%%%%%%%%%%%%%%%黑尺%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if P > 0.2       
    M=zeros(m+1);
    V=zeros(m+1);
    ROI_height=0;
    ROI_max=0;
    F=1;
    count=0;
    Thresh = 5;   %刻度区行数阈值   可调整  
    for i=1:m
        for j=1:n
            if Img_BW1(i,j)==1
                V(i) = V(i)+1;
            end
        end
        M(i) = V(i)/n;
    end  

   %将ROI缩小至刻度线所在区域
    for i=1:1:m
        if  M(i)>0.25 && M(i)<0.65 && F==1    %M（i）代表该行白色像素点所占比例 如果在某范围内（）就认为该行是刻度
            count = count + 1;
        else
            count = 0;
        end
    
        if (count > Thresh) && ((M(i+1)<0.25) || (M(i+1)>0.65)) && M(i)>0.25 && M(i)<0.65 
           F=0;      %标志位置0 即一旦找到符合要求的刻度区域就立即停止搜寻
           ROI_max = i ;
           ROI_height = count;
        end
    end 
    Img_ROI = imcrop(Img_BW1,[ceil(0.2*n), ROI_max-ROI_height,ceil(0.6*n),ROI_height]); 
    %figure,imshow(Img_ROI),title('Img ROI');

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %为了提高后续检测精度  对刻度线进行去毛刺和补缺损处理 
  %去毛刺 补缺损 结构元素法
  %去毛刺：搜索每列 如果有连续白色像素且个数小于N 则认为是毛刺 将其灰度值设为黑色 即去除毛刺
  %
  %在去毛刺基础上 补缺损 方法与去毛刺类似
  %补缺损：搜索每列 如果有连续黑色像素且个数小于N 则认为是缺损（孔洞） 将其灰度值设为白色 即补缺损
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %去毛刺
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

 %补缺损
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
  %利用最小二乘法 计算图中刻度尺1cm距离内所包含的像素个数 a0
  %并从增加样本基数 剔除异常数据 增加样本间隔三个方面减小误差 提高精度
  %1、增加样本基数： 要分情况处理：对于较宽的刻度尺 可取（2d+1）行数据；对较窄的刻度尺 只取最中间行数据
  %2、剔除异常数据： 在原始样本基础上通过最小二乘法初步估计出一个刻度线间隔所包含的像素数 以此为参考剔除异常数据 
  %3、增大样本间隔： 计算10个刻度，即1cm间隔内的像素个数，从1mm提高到1cm
  %
  % H数组存放每个刻度线的位置数据
  % G数组存放各刻度线间隔内的像素点数
  % R数组内存放剔除异常数据后刻度线间隔内的像素个数 以mm为单位
  % S数组内存放剔除异常数据后10个刻度线间隔内的像素个数 以cm为单位
  % T数组内存放S数组中数据转化得到的位置信息 便于最小二乘法计算
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   [m,n] = size(Img_ROI);  
   d=3; 
   %减小误差方法1 多取几行 增加样本基数
   if ceil(ROI_height/2)>d   
      H = zeros(2*d+1,n);     %H数组存放每个刻度线的位置
      G = zeros();            %G数组存放各刻度线间隔内的像素点数
      E = zeros(); 
      h=0;
      f=0;
      for i=(ceil(ROI_height/2)-d):(ceil(ROI_height/2)+d)
          h=h+1;%行数
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
          E(h) = k;%E数组存放各行刻度线个数
      end
      
 %通过最小二乘法初步估计出一个刻度线间隔所包含的像素数 以此为参考剔除异常数据      
 x=[1:1:E(d+1)-1];
 y=H(d+1,2:1:E(d+1));
 p=polyfit(x,y,1);%最小二乘拟合
 a1=polyval(p,1);
 a2=polyval(p,2);
 a=a2-a1;
 
 %减小误差方法2 剔除异常数据
 R = zeros();   %R数组内存放剔除异常数据后刻度线间隔内的像素个数 以mm为单位
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
 
   %%%%只取最中间行数据进行计算     
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
 p=polyfit(x,y,1);  %最小二乘法拟合
 a1=polyval(p,1);
 a2=polyval(p,2);
 a=a2-a1;
    
  %剔除异常数据
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
%%%%%%%%%减小误差方法3 增大样本间隔 计算10个刻度，即1cm间隔内的像素个数
  S = zeros(n);    %S数组内存放剔除异常数据后10个刻度线间隔内的像素个数 以cm为单位
  s=1;
  for i=1:r-9
      for j=0:9
          S(s)=S(s)+R(i+j);
      end
      s=s+1;
  end
   
T = zeros(n); %转化为位置信息 便于最小二乘法计算
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%白尺%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else
 level=0.8; %调高阈值重新二值化
 Img_BW2 =im2bw(Img_rotate2,level);
 %figure,imshow(Img_BW2);
    
    %提取刻度线区域
    M=zeros(m);
    N=zeros(m);
    ROI_height=0;
    ROI_max=0;
    F=1;
    count=0;
    Thresh =3;   %可调整
    for i=1:m
        for j=1:n
            if Img_BW2(i,j)==1
                N(i) = N(i)+1;
            end
        end
        M(i) = N(i)/n;
    end
    
    for i=1:m
        if  M(i)>0.3 && M(i)<0.7 && F==1    %M（i）代表该行白色像素点所占比例 如果在某范围内（）就认为是刻度线区域
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
  %  figure,imshow(Img_ROI);   %ROI为灰度图像

   [m,n] = size(Img_ROI);  
    d=1; 
   
   %减小误差手段1 多取几行 增加最小二乘法样本数据基数
   if ceil(ROI_height/2)>d   
      H = zeros(2*d+1,n);     %每个刻度线的位置
      G = zeros();            %各刻度线之间的像素点数
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

 %通过最小二乘法初步估计出一个刻度线间隔所包含的像素数 以此为参考剔除异常数据    
 x=[1:1:E(d+1)-1];
 y=H(d+1,2:1:E(d+1));
 p=polyfit(x,y,1);
 a1=polyval(p,1);
 a2=polyval(p,2);
 a=a2-a1;
 
 %减小误差手段2 剔除异常数据
 R = zeros();   %剔除异常数据后刻度线间像素个数 以mm为单位
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
     %%%%只取最中间行数据进行计算  
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

  %通过最小二乘法初步估计出一个刻度线间隔所包含的像素数 以此为参考剔除异常数据  
  x=[1:1:k];
  y=H(1:1:k);
  p=polyfit(x,y,1);
  a1=polyval(p,1);
  a2=polyval(p,2);
  a=a2-a1;
  
   %减小误差手段2 剔除异常数据
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
%%%%%%%%%减小误差手段3 计算10个刻度，即1cm之间的像素个数
 S = zeros(n);    %剔除异常数据后10个刻度线间像素个数 以cm为单位
  s=1;
  for i=1:r-9
      for j=0:9
          S(s)=S(s)+R(i+j);
      end
      s=s+1;
  end
   
T = zeros(n); %转化为位置信息 便于最小二乘法计算
t=1;
T(1)=0;
for i=1:s 
    t=t+1;
    T(t)=T(t-1)+S(i);
end

%最小二乘拟合   
x=[1:1:t];
y=T(1:1:t);
p=polyfit(x,y,1);
a1=polyval(p,1);
a2=polyval(p,2);
a0=a2-a1;          
%plot(x,y,'*');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%第三部分%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%第二部分得到图中尺子10个刻度线间隔内 即1cm包含的像素数
%实际距离与像素的换算关系为 72像素/1英寸 即28.3464578像素/cm
%确定比例尺 即图像像素距离与实际像素距离之间的比例关系
%利用比例尺将图像缩放到实际大小
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%缩放比例尺
scale=28.3464578/a0;  
real_Img=imresize(Img,scale,'bilinear');
%figure,imshow(real_Img);
imwrite(real_Img,['C:\Users\pc\Desktop\足迹\实际大小足迹图像\real (',num2str(c),').bmp'],'bmp');%存储实际大小足迹图像的路径


end

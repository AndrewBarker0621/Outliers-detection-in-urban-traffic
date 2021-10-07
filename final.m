filename1='20111115.csv'
V1= csvread(filename1,0,3)
filename2='20111116.csv'
V2= csvread(filename2,0,3)
filename3='20111117.csv'
V3= csvread(filename3,0,3)
filename4='20111118.csv'
V4= csvread(filename4,0,3)
filename5='20111121.csv'
V5= csvread(filename5,0,3)
filename6='20111122.csv'
V6= csvread(filename6,0,3)
filename7='20111123.csv'
V7= csvread(filename7,0,3)
filename8='20111124.csv'
V8= csvread(filename8,0,3)
filename9='20111125.csv'
V9= csvread(filename9,0,3)
filename10='20111128.csv'
V10= csvread(filename10,0,3)
V=[V1,V2,V3,V4,V5,V6,V7,V8,V9,V10]


% V是一个含有n个样本的矩阵，每一列对应一个样本，每个样本为m维
[m,n] = size(V);

r=7;

% 随机初始化W和H矩阵
W = rand(m,r);
H = rand(r,n);

% 设置最大迭代次数，保证分解结果是收敛的
count = 5000;
eps = 1000;
errs = zeros(count,1);

for t = 1:count

   W = W .* ( (V*H') ./ max(W*(H*H'), eps) ); 
   H = H .* ( (W'*V) ./ max((W'*W)*H, eps) );

   loss = sum((V-W*H).^2);
   errs(t) = sum(sum(loss));
end

plot(H')
grid on
 ylabel('Relative Traffic Volume')
 xlabel('Time in Hour')
 
% K: 表示将W划分为几类，为整数
% Idx: N*1的向量，存储的是每个点的聚类标号
% C: K*P的矩阵，存储的是K个聚类质心位置
% sumD: 1*K的和向量，存储的是类间所有点与该类质心点距离之和
% D: N*K的矩阵，存储的是每个点与所有质心的距离

 opts = statset('Display','final');
[Idx,C,sumD,D]=kmeans(W,300,'Replicates',300,'Options',opts,'Maxiter',1000);

plot3(C(:,1),C(:,2),C(:,3),'kx','MarkerSize',2,'LineWidth',1.5)
hold on
grid on

%plot3(CC(:,1),CC(:,2),CC(:,3),'kx','MarkerSize',2,'LineWidth',2)
%hold on
%grid on

for i=1:1:300
    plot3(W(Idx==i,1),W(Idx==i,2),W(Idx==i,3),'.','MarkerSize',2)
    xlabel('coefficient A', 'Fontsize',12)
    hold on
    ylabel('coefficient B', 'Fontsize',12)
    hold on
    zlabel('coefficient C', 'Fontsize',12)
    hold on
    grid on 
end

filename11='20111129.csv'
V11= csvread(filename11,0,3)

%取得历史数据的均值
Vm=rand(127049,96);
for i=1:1:127049
    for j=1:1:96
            Vm(i,j)=[mean(V(i,j))];
    end
end

%计算历史数据的泊松分布
px=poisscdf(V11,Vm);

for j=1:1:96
    F(1,j)=j;
end

% csvwrite('Idx.csv',Idx)
% csvwrite('past.csv',px)
% csvwrite('neighbor.csv',NMP)

m=1;
for j=1:1:127049
    if Idx(j,1)==3
        FF3(m,:)=V11(j,:);
        m=m+1;
    end
end

dfittool(FF3m)

% for i=1:1:5
%    m=1;
%    for j=1:1:127049
%        if Idx(j,1)==1
%            F(i)(m,:)=V11(j,:);
%         m=m+1;
%        end
%    end
% end

% m=1;
% for j=1:1:127049
%     if Idx(j,1)==3
%         FF3(m,:)=V11(j,:);
%         m=m+1;
%     end
% end
% for j=1:1:96
%     FF3m(1,j)=[mean(FF3(:,j))];
% end
% dfittool(FF3m)
 
% eval(['F' num2str(i) '(m,:)' '=' 'V11' '(j,:)' ]);


for i = 1:300
    Idx2 = find(Idx == i);
    num = size(Idx2,1);
    for j=1:96
      n2=0;
      for l=1:num  
        n2=n2+V11(Idx2(l),j);
      end
      
      for k=1:num
          n2=n2-V11(Idx2(k),j);
          average=n2/(num-1); 
          NMP(Idx2(k),j)=poisspdf(V11(Idx2(k),j),average) ;
      end 
    end
end
% 
% filename11= 'beijing.csv'
% exceptional= csvread(filename11)

for i=1:1:96
    for j=1:1:127049
        F(j,i)=0;
    end
end

file1= 'ab21.csv'
F1= csvread(file1,0,2)
file2= 'ab22.csv'
F2= csvread(file2,0,2)
file3= 'ab23.csv'
F3= csvread(file3,0,2)
file4= 'ab24.csv'
F4= csvread(file4,0,2)
file5= 'ab25.csv'
F5= csvread(file5,0,2)
file8= 'ab28.csv'
F8= csvread(file8,0,2)
file9= 'ab29.csv'
F9= csvread(file9,0,2)

P=[px';NMP']
T=[F1']
px1=poisscdf(V1,Vm);
P1=[px1';NMP']


[input,ps1]=mapminmax(P);
[target,ps2]=mapminmax(T); 
%创建网络
net=newff(input,target,12,{'tansig','purelin'},'trainlm');
%设置训练次数
net.trainParam.epochs=1000;
%设置收敛误差
net.trainParam.goal=0.0001;
LP.lr=0.0001;
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
%训练网络
net=train(net,input,target);
%将输入数据归一化
ainput=mapminmax('apply',P1,ps1);
%放入到网络输出数据
boutput=net(ainput);
%将得到的数据反归一化得到预测数据
prediction=mapminmax('reverse',boutput,ps2);

%判断整体的查全率
coutput=boutput'
coutput=-1*coutput
recall(1,1)=0
for i=1:1:127049
    for j=1:1:24
        if coutput(i,j)==F1(i,j)
            recall(1,1)=recall(1,1)+1
        end
    end
end
recall(1,2)=127049*24
recall(1,3)=recall(1,1)/recall(1,2)



%设置阈值
coutput=-1*boutput'
for i=1:1:127049
    for j=1:1:24
        if coutput(i,j)<=1-0.0003 || coutput(i,j)>=1+0.0003
            coutput(i,j)=0;
        else
            coutput(i,j)=1;
        end
    end
end

%判断异常预测的查全率
recall(1,1)=0
for i=1:1:127049
    for j=1:1:24
        if coutput(i,j)==F1(i,j) && F1(i,j)==1
            recall(1,1)=recall(1,1)+1
        end
    end
end
recall(1,2)=0
for i=1:1:127049
    for j=1:1:24
        if F1(i,j)==1
            recall(1,2)=recall(1,2)+1
        end
    end
end
recall(1,3)=recall(1,1)/recall(1,2)


precision=0
%设置异常预测准确率
precision(1,1)=0
for i=1:1:127049
    for j=1:1:24
        if coutput(i,j)==F1(i,j) && F1(i,j)==1
            precision(1,1)=precision(1,1)+1
        end
    end
end
precision(1,2)=0
for i=1:1:127049
    for j=1:1:24
        if coutput(i,j)==1
            precision(1,2)=precision(1,2)+1
        end
    end
end
precision(1,3)=precision(1,1)/precision(1,2)

fmeasure(1,1)=(2*precision(1,3)*recall(1,3))/(precision(1,3)+recall(1,3));


threshold(1,1)=0.002
for i=2:1:15
    threshold(i,1)=threshold(i-1,1)+0.001
end

for i=1:1:15
    recall(i,3)=recall(i,1)/recall(i,2)
    precision(i,3)=precision(i,1)/precision(i,2)
    fmeasure(i,1)=(2*precision(i,3)*recall(i,3))/(precision(i,3)+recall(i,3));
end

for i=1:1:15
    final(i,1)=recall(i,3);
    final(i,2)=precision(i,3);
    final(i,3)=fmeasure(i,1);
end
plot(final)
grid on
axis([1 15 0.16 0.44])






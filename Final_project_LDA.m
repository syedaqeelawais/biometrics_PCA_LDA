clc
clear

%laod the database
database=load('ORL_32x32.mat');
database2=load('1.mat');
%wxtract train and testing data
fea_train = database.fea(database2.trainIdx,:); 
fea_test = database.fea(database2.testIdx,:);
%convert it column vectors
%fea_train=fea_train';
%fea_test=fea_test';
%showing the 10 pictures in databse
%there are 40 distinct images
faceW = 32; 
faceH = 32; 
numPerLine = 40; 
ShowLine = 10; 

Y = zeros(faceH*ShowLine,faceW*numPerLine); 
for i=0:ShowLine-1 
  	for j=0:numPerLine-1 
    	Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(database.fea(i*numPerLine+j+1,:),[faceH,faceW]); 
  	end 
end 
imagesc(Y);colormap(gray);
%%
%Extracting mean all the training vetors

dataMean=mean(fea_train,1);
%me=mean(fea_train,2);
%because the training set is taken using two images
%of the one one subject so considering the mean of 
%first 2 subjects of every sample
%meu1=mean(x1) where x1 is first 2 vectors of training img
%meu2=mean(x2)
%for i=1:2:size(me,1)
 %   meu(i,:)=me(i:i+1,:);
  %  meu( all(~meu,2), : ) = [];%remove zero rows
%end

%finding the covarinace of each class obtained means 
%s1=cov(x1) where x1=two column vectors(represent class 1) of train as 2 imgs taken for
%training
%for i=1:2:size(fea_train,2)
 %   s=(cov(fea_train(:,1:size(fea_train,2))));
    %s_1=s(s~=0);
    %s( all(~s,2), : ) = [];
    %s( :, all(~s,1) ) = [];%remove zero columns
%end
%find the average face
A=zeros(size(fea_train,1),size(fea_train,2));
me = mean(fea_train,1); %mean of all the column vectors
%normalize data by subtracting mean
for i=1:size(fea_train,1)
A(i,:)=fea_train(i,:)-me;
end
avg_face=reshape(me,[32 32]);
figure
imagesc(avg_face);colormap(gray);
title('Average Face');
%%
%within class scatter matrix s_w
%for i=1:size(s,1)
 %   sw(i,:)=sum(s(i,:));%row wise sum of all the scatter matrices
%end
%between class matrix
%for i=1:size(meu,1)
    %if(i<512)
   % s_1=minus(meu(i,:),meu(i+1,:));
  %  s_2=(minus(meu(i,:),meu(i+1,:)))';
 %   end
%end
%between class matrix sb
%sb=diag(s_2*s_1);
train_label = database.gnd(database2.trainIdx); 
test_label = database.gnd(database2.testIdx);

nc=unique(train_label);
[nPoints nDims]=size(fea_train);
Sw=zeros(nDims,nDims);
Sb=zeros(nDims,nDims);
for i=1:size(nc,1)
    
    %cur_x(2,:)=fea_train(train_label==i,:);
  if size(train_label,2) == 1 % if labels are given as 1,2,3,4...
    ind = find(train_label==i);
  else % labels are 0 0 1 0 0 ...
    ind = find(train_label(:,i)==1);
  end
  
  if numel(ind) == 0  % empty class: ignore
    continue;
  end
  
  classMean = mean(fea_train(ind,:));
  
  Sw = Sw + cov(fea_train(ind,:),1);
  Sb = Sb + numel(ind)*(classMean-dataMean)'*(classMean-dataMean);
    
end
%%
%finding Fisher vectors
 inv_sw=inv(Sw);
 inv_sw_by_sb=inv_sw\Sb;
[W LAMBDA]=eig(inv_sw_by_sb);
% 
lambda=diag(LAMBDA);
[lambda, SortOrder]=sort(lambda,'descend');
W=W(:,SortOrder);
W=real(W);
% Y=fea_train*W;
% 
% % Number of principal components
p=W(:,1:10);
 % Project faces with reduce dimensions
train_w=p'*(A'- repmat(me',1,80));
lambda_1=lambda(1:10);
% reconstruct images
recon_img=p*train_w+repmat(me',1,80);
 
figure
plot(lambda,'o-'); title('lambda strength');
xlabel('Image Column vectors');
ylabel('Number of Eigen Values');
%%
 %plot top fisher faces
 figure
 top=10;
 for i=1:top
     eig_face_img = reshape(W(:,i),32,32);
     %eig_face_img=eig_face_img';
     eig_face_img=histeq(eig_face_img,255);
     subplot(ceil(sqrt(top)),ceil(sqrt(top)),i);
     imshow(eig_face_img);    
 end
 
figure
 top_1=10;
 for i=1:top_1
     eig_face_img_1 = reshape(recon_img(:,i),32,32);
     %eig_face_img=eig_face_img';
     eig_face_img_1=histeq(eig_face_img_1,255);
     subplot(ceil(sqrt(top_1)),ceil(sqrt(top_1)),i);
     imshow(eig_face_img_1);
 end
%%
% %project test images into Eigen space of PCA
A2=zeros(size(fea_test,1),size(fea_test,2));
me = mean(fea_test,1); %mean of all the column vectors
for i=1:size(fea_test,1)
A2(i,:)=fea_test(i,:)-me;
end
test_w=p'*(A2'- repmat(me',1,320));
%%
%Euclidean Distance between training and testing images
theta=275;
euclid_dist=zeros(1,80);
for i=1:size(fea_train,1)
    euclid_dist(:,i)=(norm(test_w(:,i)-train_w(:,i))); 
    match_index=find(euclid_dist<theta);
    train_data=A';
    test_data=A2';
    train_match=train_data(:,match_index);
    test_match=test_data(:,match_index);
end
%min_value=min(euclid_dist);
figure('Name','Testing input Images','NumberTitle','off')
f_size=size(match_index,2);

for j=1:f_size
  
    subplot(ceil(sqrt(f_size)),ceil(sqrt(f_size)),j);
    imagesc(reshape(test_match(:,j),32,32));colormap(gray);
end
figure('Name','Training Output Images','NumberTitle','off')
for j=1:f_size
    subplot(ceil(sqrt(f_size)),ceil(sqrt(f_size)),j);
    imagesc(reshape(train_match(:,j),32,32));colormap(gray);
    
end
%%
%have to find for eigen coefficients
%Computing Genuine score and Impostor Score
%Select top 10 eigen values
eig_face_top=W(:,1:10);
complete_data=(database.fea)';
%compute coefficients of complete database with reduced dim
eig_coeff=zeros(10,400);
for i=1:size(eig_coeff,2)
    eig_coeff(:,i)=eig_face_top'*complete_data(:,i);%As the coefficient is w=E^t* x and eigen vectors are 10
end
%Computing Genuine score
for i=1:10:size(eig_coeff,2)
    for j=i:i+9
        if(i<=400)
        value(i,j)=norm(eig_coeff(:,i)-eig_coeff(:,j));
        
        end
    end
    
end
genuine_matrix=value(value~=0);
genuine_matrix=value(value~=0);
fid = fopen('D:\Biometric\project\ldagenuine.txt','wt');
for ii = 1:size(genuine_matrix,1)
    fprintf(fid,'%g\t',genuine_matrix(ii,:));
    fprintf(fid,'\n');
end
fclose(fid)
%Imppostor Score
for i=1:size(eig_coeff,2)
    for j=i+10
        if(j<=399)
        value_1(i,j)=norm(eig_coeff(:,i)-eig_coeff(:,j));
        
        end
    end
    
end
impostor_matrix=value_1(value_1~=0);

fid_1 = fopen('D:\Biometric\project\ldaimpostor.txt','wt');
for ii = 1:size(impostor_matrix,1)
    fprintf(fid_1,'%g\t',impostor_matrix(ii,:));
    fprintf(fid_1,'\n');
end
fclose(fid_1)
%%
%Draw ROC
drawROC('D:\Biometric\project\ldagenuine.txt','D:\Biometric\project\ldaimpostor.txt','d');

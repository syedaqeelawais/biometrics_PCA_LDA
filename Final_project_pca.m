%%
clc
clear

%laod the database
database=load('ORL_32x32.mat');
database2=load('1.mat');
%wxtract train and testing data
fea_train = database.fea(database2.trainIdx,:); 
fea_test = database.fea(database2.testIdx,:);
%convert it column vectors
fea_train=fea_train';
fea_test=fea_test';
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
%find the average face
A=zeros(size(fea_train,1),size(fea_train,2));
me = mean(fea_train,2); %mean of all the column vectors
for i=1:size(fea_train,2)
A(:,i)=fea_train(:,i)-me;
end
avg_face=reshape(me,[32 32]);
figure
imagesc(avg_face);colormap(gray);
title('Average Face');
%%
[U E V]=svd(A,0);%singular value decomposition to calculate
%eig vector and eig values
eig_vec=U;
eig_value=diag(E);

%Number of principal components
p=eig_vec(:,1:10);
lambda=eig_value(1:10);
%Project each face in training set into eig space
%with reduced dimensions
train_w=p'*(A- repmat(me,1,80));
%getting reconstructed images
recon_img=p*train_w+repmat(me,1,80);

figure
plot(eig_value,'o-'); title('lambda strength');
xlabel('Image Column vectors');
ylabel('Number of Eigen Values');

%plot top Eig faces
figure
top=10;
for i=1:top
    eig_face_img = reshape(eig_vec(:,i),32,32);
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
%project test images into Eigen space of PCA
A2=zeros(size(fea_test,1),size(fea_test,2));
me = mean(fea_test,2); %mean of all the column vectors
for i=1:size(fea_test,2)
A2(:,i)=fea_test(:,i)-me;
end
test_w=p'*(A2- repmat(me,1,320));
%%
%Euclidean distance between train and testing images
theta=800;
euclid_dist=zeros(1,80);
for i=1:size(eig_vec,2)
    euclid_dist(:,i)=(norm(test_w(:,i)-train_w(:,i)));
    match_index=find(euclid_dist<theta);
    train_match=A(:,match_index);
    test_match=A2(:,match_index);
    
end
figure('Name','Testing input Images','NumberTitle','off')
f_size=size(match_index,2);
%index=reshape(1:18,2,9)';
for j=1:f_size
    %subplot(2,2,1); subplot(2,2,3);
    subplot(ceil(sqrt(f_size)),ceil(sqrt(f_size)),j);
    imagesc(reshape(test_match(:,j),32,32));colormap(gray);
    %subplot(ceil(sqrt(f_size)),ceil(sqrt(f_size)),j);
    %imagesc(reshape(train_match(:,j),32,32));
end
figure('Name','Training Output Images','NumberTitle','off')
for j=1:f_size
    subplot(ceil(sqrt(f_size)),ceil(sqrt(f_size)),j);
    imagesc(reshape(train_match(:,j),32,32));colormap(gray);
    
end

%[euclid_dist_min recognized_index] = min(euclid_dist);
%recognized_img = strcat(int2str(recognized_index),'.jpg');


%match_imgs = strcat(int2str(match_index),'.jpg');
%ss=eig_vec(:,match_index);
%%
%have to find for eigen coefficients
%Computing Genuine score and Impostor Score
%Select top 10 eigen values
eig_face_top=eig_vec(:,1:10);
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
fid = fopen('D:\Biometric\project\mygenuine.txt','wt');
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

fid_1 = fopen('D:\Biometric\project\myimpostor.txt','wt');
for ii = 1:size(impostor_matrix,1)
    fprintf(fid_1,'%g\t',impostor_matrix(ii,:));
    fprintf(fid_1,'\n');
end
fclose(fid_1)
%%
%Draw ROC
drawROC('D:\Biometric\project\mygenuine.txt','D:\Biometric\project\myimpostor.txt','d');

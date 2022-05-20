clear all;clc;
addpath(genpath('..\CTV-SPCP\'))
%% load data
hsi_name = 'pure_DCmall';
load([hsi_name,'.mat'])
clean_data       = Ori_H;
clean_data       = Normalize(clean_data);
[M,N,p]        = size(clean_data);
gaussian_level = 0.2;
sparse_level   = 0.2;
noise_data       = GetNoise(clean_data,gaussian_level,sparse_level);
D = reshape(noise_data,[M*N,p]);
Result = cell(4,1);
Result{1}=clean_data;
Result{2}=noise_data;
[mpsnr(2),mssim(2),ergas(2)]=msqia(clean_data, noise_data);
%% sqrt-ctv_spcp
it =3;
fprintf('======== sqrt-ctv-pcp  ========\n')
[rec_tensor,~] = ctv_sqrt_spcp(noise_data);
[mpsnr(it),mssim(it),ergas(it)]=msqia(clean_data, rec_tensor);
Result{it}=rec_tensor;
%% sqrt-pcp
it = 4;
fprintf('======== sqrt-pcp  ========\n')
[A_hat,E_hat,iter] = spcp_sqrt(D);
rec_tensor = reshape(A_hat,[M,N,p]);
[mpsnr(it),mssim(it),ergas(it)]=msqia(clean_data, rec_tensor);
Result{it}=rec_tensor;

index = 36;
figure;
Y = WindowGig(Result{1}(:,:,index),[0.57,0.03],[0.1,0.2],2.5,0);
subplot(2,2,1);imshow(Y,[]);title('original band')
Y = WindowGig(Result{2}(:,:,index),[0.57,0.03],[0.1,0.2],2.5,0);
subplot(2,2,2);imshow(Y,[]);title(['noise, psnr:',num2str(mpsnr(2))])
Y = WindowGig(Result{3}(:,:,index),[0.57,0.03],[0.1,0.2],2.5,0);
subplot(2,2,3);imshow(Y,[]);title(['ctv-sqrt-pcp, psnr:',num2str(mpsnr(3))])
Y = WindowGig(Result{4}(:,:,index),[0.57,0.03],[0.1,0.2],2.5,0);
subplot(2,2,4);imshow(Y,[]);title(['sqrt-pcp, psnr:',num2str(mpsnr(4))])
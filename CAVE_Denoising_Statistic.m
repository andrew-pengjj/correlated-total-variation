clear all;
clc;
load('fileNames.mat')
len_data = length(fileNames);
Dir = 'Data/CAVE';
gaussian_list = [0.1,0.2,0.3,0.4,0,0,0,0,0.2,0.2,0.2,0.2];
sparse_list   = [0,0,0,0,0.1,0.2,0.3,0.4,0.05,0.1,0.15,0.2];
len_noise = length(gaussian_list);
mpsnr = zeros(len_data,len_noise,10);
mssim = zeros(len_data,len_noise,10);
ergas = zeros(len_data,len_noise,10);
for data_id=1:len_data
    %% load data
    Ori_H  =  LoadCAVE(fileNames{data_id},Dir);
    fprintf('=========== data= %s =============\n',fileNames{data_id})
    clean_data       = Ori_H;
    clean_data       = Normalize(clean_data);
    [M,N,p]          = size(clean_data);
    %% denoising result
    for noise_id = 1:len_noise
        gaussian_level = gaussian_list(noise_id);
        sparse_level  = sparse_list(noise_id);
        noise_data    = GetNoise(clean_data,gaussian_level,sparse_level);
        D             = reshape(noise_data,[M*N,p]);
        %% 3dctv_spcp
        it = 1;
        fprintf('======== 3DCTV-SPCP  ========\n')
        [rec_tensor,~] = ctv_alm_spcp(noise_data,gaussian_level);%ctv_con_spcp
        [mpsnr(data_id,noise_id,it),mssim(data_id,noise_id,it),ergas(data_id,noise_id,it)]=msqia(clean_data, rec_tensor);
        %% sqrt-ctv_spcp
        it =2;
        fprintf('======== sqrt-ctv-pcp  ========\n')
        [rec_tensor,~] = ctv_sqrt_spcp(noise_data);
        [mpsnr(data_id,noise_id,it),mssim(data_id,noise_id,it),ergas(data_id,noise_id,it)]=msqia(clean_data, rec_tensor); 
        %% ctv_rpcp
        it = 3;
        fprintf('======== 3DCTV-RPCA  ========\n')
        [rec_tensor,~] =ctv_rpca(noise_data);
        [mpsnr(data_id,noise_id,it),mssim(data_id,noise_id,it),ergas(data_id,noise_id,it)]=msqia(clean_data, rec_tensor); 
        %% spcp 
        it = 4;
        fprintf('======== spcp ========\n')
        [L_hat,E] = spcp_alm(D,gaussian_level);
        L_hat_init = L_hat;
        rec_tensor = reshape(L_hat,[M,N,p]);
        [mpsnr(data_id,noise_id,it),mssim(data_id,noise_id,it),ergas(data_id,noise_id,it)]=msqia(clean_data, rec_tensor); 
        %% pq-sum
        it = 5;
        fprintf('======== p,q-spcp ========\n')
        p_value = 0.2;
        q_value = p_value;
        lambda1 = ((sqrt(M*N)+sqrt(p))*gaussian_level+1e-5);
        lambda2 = ((sqrt(M*N)+sqrt(p))*gaussian_level/sqrt(M*N)+1e-5/sqrt(M*N));
        mu1 = 1;
        mu2 = 1;
        [A_hat,~]=pqSPCP(D,L_hat_init,p_value,q_value,lambda1,lambda2,mu1,mu2);
        rec_tensor = reshape(A_hat,[M,N,p]);
        [mpsnr(data_id,noise_id,it),mssim(data_id,noise_id,it),ergas(data_id,noise_id,it)]=msqia(clean_data, rec_tensor); 
        %% sqrt-pcp
        it = 6;
        fprintf('======== sqrt-pcp  ========\n')
        [A_hat,E_hat,iter] = spcp_sqrt(D);
        rec_tensor = reshape(A_hat,[M,N,p]);
        [mpsnr(data_id,noise_id,it),mssim(data_id,noise_id,it),ergas(data_id,noise_id,it)]=msqia(clean_data, rec_tensor);
        %% pcp
        it = 7;
        fprintf('======== pcp  ========\n')
        [L_hat,~,~] = rpca(D);
        rec_tensor = reshape(L_hat,[M,N,p]);
        [mpsnr(data_id,noise_id,it),mssim(data_id,noise_id,it),ergas(data_id,noise_id,it)]=msqia(clean_data, rec_tensor);
        %% wnnm
        it = 8;
        fprintf('======== wnnm  ========\n')
        tol=1e-6;
        maxIter=100;
        C = 0.01;
        [A_hat,~,~] = inexact_alm_WNNMrpca(D,C,tol, maxIter);
        rec_tensor = reshape(A_hat,[M,N,p]);
        [mpsnr(data_id,noise_id,it),mssim(data_id,noise_id,it),ergas(data_id,noise_id,it)]=msqia(clean_data, rec_tensor);
        %% LRMR
        it = 9;
        fprintf('======== lrmr  ========\n')
        sparsity = 0.1;   % for LRMR implemented with ssGoDec
        rk = 3;
        blocksize = 20;
        stepsize  = 8;
        rec_tensor = LRMR_HSI_denoise(noise_data,rk,blocksize,sparsity,stepsize);
        [mpsnr(data_id,noise_id,it),mssim(data_id,noise_id,it),ergas(data_id,noise_id,it)]=msqia(clean_data, rec_tensor);
        %% lrtv
        it = 10;
        fprintf('======== lrtv  ========\n')
        tau     = 0.01;
        lambda  = 40/sqrt(M*N);
        rk = 3;
        rec_tensor = LRTV(noise_data, tau,lambda, rk);
        [mpsnr(data_id,noise_id,it),mssim(data_id,noise_id,it),ergas(data_id,noise_id,it)]=msqia(clean_data, rec_tensor);
    end
end
metric_result.mpsnr = mpsnr;
metric_result.mssim = mssim;
metric_result.ergas = ergas;
save_name = ['CAVE_denoising_part1.mat'];
save(save_name,'metric_result')
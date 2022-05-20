clear all;clc
addpath(genpath('..\CTV-SPCP\'))
% data_list
data_list=["airport","b0","ShoppingMall","SwitchLight","Escalator","Curtain","trees","WaterSurface","Fountain"];
weight_tv = [10,1,1,2,1,2,1,13,1];
len_data  = length(data_list);
len_method = 11;
std_list   = [0,0.02,0.04,0.05,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2];
len_std    = length(std_list);
auc_result= zeros(len_data,len_method,len_std);
Result = cell(11,1);
for data_id = 1:len_data
    for std_id = 1:len_std
        data_name = data_list(data_id);
        [original_data,gt_fore]=GetVideoMask(data_name);
        [M,N,p]=size(original_data);
        gaussian_level = std_list(std_id);
        InputTensor = GetNoise(original_data,gaussian_level,0);
        InputMatrix       = reshape(InputTensor,[M*N,p]);
        weight = weight_tv(data_id);
        %% 3dctv_spcp
        it = 1;
        fprintf('======== 3DCTV-SPCP  ========\n')
        [rec_tensor,E] = ctv_alm_spcp(InputTensor,gaussian_level,weight);%ctv_con_spcp
        E_tensor = reshape(E,[M,N,p]);
        tmp = MAUC(gt_fore,abs(InputTensor-rec_tensor));
        if tmp<=MAUC(gt_fore,abs(E_tensor))
            auc_result(data_id,it,std_id) = MAUC(gt_fore,abs(E_tensor));
            Result{it} = E_tensor;
        else
            auc_result(data_id,it,std_id) = tmp;
            Result{it} = InputTensor-rec_tensor;
        end
        %% sqrt-ctv-pcp
        fprintf('======== sqrt-ctv-pcp  ========\n')
        [rec_tensor,E] =ctv_sqrt_spcp(InputTensor,weight);
        E_tensor = reshape(E,[M,N,p]);
        it =2;
        tmp = MAUC(gt_fore,abs(InputTensor-rec_tensor));
        if tmp<=MAUC(gt_fore,abs(E_tensor))
            auc_result(data_id,it,std_id) = MAUC(gt_fore,abs(E_tensor));
            Result{it} = E_tensor;
        else
            auc_result(data_id,it,std_id) = tmp;
            Result{it} = InputTensor-rec_tensor;
        end
        %% 3dctv_rpca
        it = 3;
        fprintf('======== 3DCTV-RPCA  ========\n')
        weight = weight_tv(data_id);
        [rec_tensor,~] =ctv_rpca(InputTensor,weight);
        auc_result(data_id,it,std_id) = MAUC(gt_fore,abs(InputTensor-rec_tensor));
        Result{it} = InputTensor-rec_tensor;
        %% spcp
        it = 4;
        fprintf('======== spcp ========\n')
        [L_hat,E] = spcp_alm(InputMatrix,gaussian_level);
        L_hat_init = L_hat;
        rec_tensor = reshape(L_hat,[M,N,p]);
        E_tensor = reshape(E,[M,N,p]);
        tmp = MAUC(gt_fore,abs(InputTensor-rec_tensor));
        if tmp<=MAUC(gt_fore,abs(E_tensor))
            auc_result(data_id,it,std_id) = MAUC(gt_fore,abs(E_tensor));
            Result{it} = E_tensor;
        else
            auc_result(data_id,it,std_id) = tmp;
            Result{it} = InputTensor-rec_tensor;
        end
        %% pq-sum
        p_value = 0.2;
        q_value = p_value;
        lambda1 = ((sqrt(M*N)+sqrt(p))*gaussian_level+1e-5);
        lambda2 = ((sqrt(M*N)+sqrt(p))*gaussian_level/sqrt(M*N)+1e-5/sqrt(M*N));
        mu1 = 1;
        mu2 = 1;
        [A_hat,E_hat]=pqSPCP(InputMatrix,L_hat_init,p_value,q_value,lambda1,lambda2,mu1,mu2);
        rec_tensor = reshape(A_hat,[M,N,p]);
        E_tensor = reshape(E_hat,[M,N,p]);
        it = 5;
        tmp = MAUC(gt_fore,abs(InputTensor-rec_tensor));
        if tmp<=MAUC(gt_fore,abs(E_tensor))
            auc_result(data_id,it,std_id) = MAUC(gt_fore,abs(E_tensor));
            Result{it} = E_tensor;
        else
            auc_result(data_id,it,std_id) = tmp;
            Result{it} = InputTensor-rec_tensor;
        end
        %% sqrt-pcp
        fprintf('======== sqrt-pcp  ========\n')
        [A_hat,E_hat,iter] = spcp_sqrt(InputMatrix);
        rec_tensor = reshape(A_hat,[M,N,p]);
        E_tensor = reshape(E_hat,[M,N,p]);
        it = 6;
        tmp = MAUC(gt_fore,abs(InputTensor-rec_tensor));
        if tmp<=MAUC(gt_fore,abs(E_tensor))
            auc_result(data_id,it,std_id) = MAUC(gt_fore,abs(E_tensor));
            Result{it} = E_tensor;
        else
            auc_result(data_id,it,std_id) = tmp;
            Result{it} = InputTensor-rec_tensor;
        end
        %% pcp
        it = 7;
        fprintf('======== pcp ========\n')
        [L_hat,~,~] = rpca(InputMatrix);
        rec_tensor = reshape(L_hat,[M,N,p]);
        Result{it} = InputTensor-rec_tensor;
        auc_result(data_id,it,std_id) = MAUC(gt_fore,abs(InputTensor-rec_tensor));
        
        %% GODEC
        it = 8;
        fprintf('======== godec========\n')
        [L_hat,S,error,time]=GreGoDec(InputMatrix,2,7,1e-3,5,1);
        rec_tensor         = reshape(L_hat,[M,N,p]);
        auc_result(data_id,it,std_id) = MAUC(gt_fore,abs(InputTensor-rec_tensor));
        Result{it} = InputTensor-rec_tensor;
        %% DECOLOR
        it = 9;
        fprintf('======== decolor  ========\n')
        [L_hat,S_hat]   = DECOLOR(InputMatrix);
        rec_tensor      = reshape(L_hat,[M,N,p]);
        auc_result(data_id,it,std_id) = MAUC(gt_fore,abs(InputTensor-rec_tensor));
        Result{it} = InputTensor-rec_tensor;
        %% OMoGMF
        it =10;
        fprintf('======== OMoGMF  ========\n')
        r = 2;
        k = 3;
        [model]  =warmstart(InputMatrix,r,k);
        model.N=50*size(InputMatrix,1);model.ro=0.99; 
        model.tv.mod=0;model.imgsize=[M,N];model.tv.lamda=1;
        [L_hat,E,F,label,~]= OMoGMF(model,InputMatrix,3); % F is TV 
        tmp = MAUC(gt_fore,abs(reshape(E,[M,N,p])));
        if tmp >=MAUC(gt_fore,abs(reshape(F,[M,N,p])))
            rec_tensor      = reshape(InputMatrix-E,[M,N,p]);
        else
            rec_tensor      = reshape(InputMatrix-F,[M,N,p]);
        end
        auc_result(data_id,it,std_id) = MAUC(gt_fore,abs(InputTensor-rec_tensor));
        Result{it} = InputTensor-rec_tensor;
        %% PRMF
        it = 11;
        fprintf('======== PRMF  ========\n')
        lambdaU = 1;
        lambdaV = 1;
        tol     = 1e-2;
        rk      = 1;
        [UMatrix,VMatrix] = RPMF(InputMatrix, rk, lambdaU, lambdaV, tol);
        rec_tensor = reshape(UMatrix*VMatrix,[M,N,p]);
        auc_result(data_id,it,std_id) = MAUC(gt_fore,abs(InputTensor-rec_tensor));
        Result{it} = InputTensor-rec_tensor;    
    end
end

index = 4;
figure;subplot(2,4,1);imshow(original_data(:,:,index),[]);
subplot(2,4,2);imshow(InputTensor(:,:,index),[]);
subplot(2,4,3);imshow(abs(gt_fore(:,:,index)),[]);
subplot(2,4,4);imshow(abs(Result{2}(:,:,index)),[]);
subplot(2,4,5);imshow(abs(Result{5}(:,:,index)),[]);
subplot(2,4,6);imshow(abs(Result{6}(:,:,index)),[]);
subplot(2,4,7);imshow(abs(Result{8}(:,:,index)),[]);
subplot(2,4,8);imshow(abs(Result{10}(:,:,index)),[]);
clear all;clc
addpath(genpath('..\CTV-SPCP\'))
% data_list
data_name="airport";
[original_data,gt_fore]=GetVideoMask(data_name);
[M,N,p]=size(original_data);
gaussian_level = 0.05;
InputTensor = GetNoise(original_data,gaussian_level,0);
InputMatrix       = reshape(InputTensor,[M*N,p]);
Result = cell(3,1);
%% 3dctv_spcp
it = 1;
weight = 10;
fprintf('======== ctv-spcp  ========\n')
[rec_tensor,E] = ctv_alm_spcp(InputTensor,gaussian_level,weight);
E_tensor = reshape(E,[M,N,p]);
tmp = MAUC(gt_fore,abs(InputTensor-rec_tensor));
if tmp<=MAUC(gt_fore,abs(E_tensor))
    auc(it) = MAUC(gt_fore,abs(E_tensor));
    Result{it} = E_tensor;
else
    auc(it) = tmp;
    Result{it} = InputTensor-rec_tensor;
end
%% sqrt-ctv-pcp
fprintf('======== sqrt-ctv-pcp  ========\n')
[rec_tensor,E] =ctv_sqrt_spcp(InputTensor,weight);
E_tensor = reshape(E,[M,N,p]);
it =2;
tmp = MAUC(gt_fore,abs(InputTensor-rec_tensor));
if tmp<=MAUC(gt_fore,abs(E_tensor))
    auc(it) = MAUC(gt_fore,abs(E_tensor));
    Result{it} = E_tensor;
else
    auc(it) = tmp;
    Result{it} = InputTensor-rec_tensor;
end
%% sqrt-pcp
it =3;
fprintf('======== sqrt-pcp  ========\n')
[A_hat,E_hat,iter] = spcp_sqrt(InputMatrix);
rec_tensor = reshape(A_hat,[M,N,p]);
E_tensor = reshape(E_hat,[M,N,p]);
tmp = MAUC(gt_fore,abs(InputTensor-rec_tensor));
if tmp<=MAUC(gt_fore,abs(E_tensor))
    auc(it) = MAUC(gt_fore,abs(E_tensor));
    Result{it} = E_tensor;
else
    auc(it) = tmp;
    Result{it} = InputTensor-rec_tensor;
end

index = 4;
subplot(2,2,1);imshow(abs(gt_fore(:,:,index)),[]);title('groundtruth')
subplot(2,2,2);imshow(abs(Result{1}(:,:,index)),[]);title(['ctv-spcp:',num2str(auc(1))])
subplot(2,2,3);imshow(abs(Result{2}(:,:,index)),[]);title(['ctv-sqrt-spcp:',num2str(auc(2))])
subplot(2,2,4);imshow(abs(Result{3}(:,:,index)),[]);title(['sqrt-spcp:',num2str(auc(3))])
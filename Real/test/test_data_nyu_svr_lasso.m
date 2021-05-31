clear all
n1=49;n2=58;n3=47;

%% Load model and data
clc
load('../train/models/adhd_svm_nyu.mat')
%load('../train/models/adhd_lasso_nyu.mat')

load('./data/NYU_test_data.mat')

%% Data
responses_NYU(responses_NYU~=0)=1;

arr_spec_nyu = [];
arr_sens_nyu = [];

for exps= 1: experiments

    wr = wr_beta(:,exps);
    bias = wr_bias(exps);


    %% Test with NYU
    m=length(responses_NYU);
    Amatrix=zeros(m,n1*n2*n3);%equivalent matrix
    for i=1:m
        Amatrix(i,:) = reshape(double(data_NYU(i,:,:,:)),n1*n2*n3,1);
    end

    % Normalization
    for j=1:n1*n2*n3
        if norm(norm(Amatrix(:,j))) ~= 0
            Amatrix(:,j) = Amatrix(:,j) - mean(Amatrix(:,j));
        end
    end
    
    y_test_NYU = Amatrix*wr+bias;    
    y_test_NYU(y_test_NYU>0.5)=1;
    y_test_NYU(y_test_NYU<=0.5)=0;


    %% Compute sensitivity (TPR)
    arr_sens_nyu = [arr_sens_nyu; sum(y_test_NYU== 1 & responses_NYU== 1) / sum(responses_NYU==1)];

    %% Compute specificity (Tnyu)
    arr_spec_nyu = [arr_spec_nyu; sum(y_test_NYU== 0 & responses_NYU== 0) / sum(responses_NYU==0)];

end

median(arr_spec_nyu)
median(arr_sens_nyu)

clear all
n1=49;n2=58;n3=47;

%% Load model and data
clc
load('../train/models/adhd_svm_nr.mat')
%load('../train/models/adhd_lasso_nr.mat')

load('./data/NeuroIMAGE_test_data.mat')

%% Data
responses_NeuroIMAGE(responses_NeuroIMAGE~=0)=1;

arr_spec_NeuroIMAGE = [];
arr_sens_NeuroIMAGE = [];

for exps= 1: experiments

    wr = wr_beta(:,exps);
    bias = wr_bias(exps);

    %% Test with NeuroIMAGE
    m=length(responses_NeuroIMAGE);
    Amatrix=zeros(m,n1*n2*n3);%equivalent matrix
    for i=1:m
        Amatrix(i,:) = reshape(double(data_NeuroIMAGE(i,:,:,:)),n1*n2*n3,1);
    end

    % Normalization
    for j=1:n1*n2*n3
        if norm(norm(Amatrix(:,j))) ~= 0
            Amatrix(:,j) = Amatrix(:,j) - mean(Amatrix(:,j));
        end
    end
    
    y_test_NeuroIMAGE = Amatrix*wr+bias;    
    y_test_NeuroIMAGE(y_test_NeuroIMAGE>0.5)=1;
    y_test_NeuroIMAGE(y_test_NeuroIMAGE<=0.5)=0;


    %% Compute sensitivity (TPR)
    arr_sens_NeuroIMAGE = [arr_sens_NeuroIMAGE; sum(y_test_NeuroIMAGE== 1 & responses_NeuroIMAGE== 1) / sum(responses_NeuroIMAGE==1)];

    %% Compute specificity (TNeuroIMAGE)
    arr_spec_NeuroIMAGE = [arr_spec_NeuroIMAGE; sum(y_test_NeuroIMAGE== 0 & responses_NeuroIMAGE== 0) / sum(responses_NeuroIMAGE==0)];

end

median(arr_spec_NeuroIMAGE)
median(arr_sens_NeuroIMAGE)

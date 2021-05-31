clear all
n1=49;n2=58;n3=47;

%% Load model and data
clc
%load('../train/models/adhd_cp_kki.mat')
%load('../train/models/adhd_tucker_kki.mat')
load('../train/models/adhd_tpgd_kki.mat')

load('./data/KKI_test_data.mat')

%% Data
responses_KKI(responses_KKI~=0)=1;

arr_spec_nr = [];
arr_sens_nr = [];

for exps= 1: experiments

    w = wr_tensor(:,exps);

    %% Test with KKI
    m=length(responses_KKI);
    Amatrix=zeros(m,n1*n2*n3);%equivalent matrix
    for i=1:m
        Amatrix(i,:) = reshape(double(data_KKI(i,:,:,:)),n1*n2*n3,1);
    end

    % Normalization
    for j=1:n1*n2*n3
        if norm(norm(Amatrix(:,j))) ~= 0
            Amatrix(:,j) = Amatrix(:,j) - mean(Amatrix(:,j));
        end
    end

    y_test_KKI = Amatrix*w;
    y_test_KKI(y_test_KKI>0.5)=1;
    y_test_KKI(y_test_KKI<=0.5)=0;


    %% Compute sensitivity (TPR)
    arr_sens_nr = [arr_sens_nr; sum(y_test_KKI== 1 & responses_KKI== 1) / sum(responses_KKI==1)];

    %% Compute specificity (TNR)
    arr_spec_nr = [arr_spec_nr; sum(y_test_KKI== 0 & responses_KKI== 0) / sum(responses_KKI==0)];

end

median(arr_spec_nr)
median(arr_sens_nr)

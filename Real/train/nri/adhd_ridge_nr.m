%% Load packages
clear all; clc;
addpath(genpath('../../../tensor_toolbox'))
addpath(genpath('../../../tensorlab_2016-03-28'))
addpath(genpath('../../../SparsePCA_V2_0'))
addpath(genpath('../../../modules'))

%% Load preprocessed data
load('../data/NeuroIMAGE_data.mat')

rng(1)
%% Initialization/Parameters
n1 = 49; n2 = 58; n3 = 47;
experiments = 50;

%% Generate response vector, data tensor, and initialize unknown parameter tensor

% generating response vector
y = [responses_arr_1];

% Labels 1,2,and 3 get mapped to 1
y(y~=0) = 1;
% Sample size
m = length(y);

% generating data tensor
A=zeros(m,n1,n2,n3);
i=1;

for j=1:length(responses_arr_1)
        A(i,:,:,:)=data_arr_1(j,:,:,:);
        i=i+1;
end

% clear loaded data
clear data_arr_1
clear responses_arr_1

% Equivalent matrix
Amatrix=zeros(m,n1*n2*n3);
for i = 1:m
    Amatrix(i,:) = reshape(double(A(i,:,:,:)),n1*n2*n3,1);
end

% Normalization
for j=1:n1*n2*n3
    if norm(norm(Amatrix(:,j))) ~= 0
        Amatrix(:,j) = Amatrix(:,j) - mean(Amatrix(:,j));

    end
end

wr_beta =[];
wr_bias =[];
wr_lambda = [];
for exp = 1:experiments
    wr = fitrlinear(Amatrix,y,'Learner','leastsquares','Regularization','ridge');
    wr_beta = [wr_beta wr.Beta];
    wr_bias = [wr_bias wr.Bias];
    wr_lambda = [wr_lambda wr.Lambda];    
end

save('adhd_ridge_nr.mat', 'experiments', 'wr_beta', 'wr_bias', ...
    'wr_lambda')
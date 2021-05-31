%% Load packages
clear all; clc;
addpath(genpath('../../../tensor_toolbox'))
addpath(genpath('../../../tensorlab_2016-03-28'))
addpath(genpath('../../../SparsePCA_V2_0'))
addpath(genpath('../../../modules'))

%% Load preprocessed data
load('../data/NYU_data.mat')

rng(1)
%% Initialization/Parameters
n1 = 49; n2 = 58; n3 = 47;
experiments = 50;
r=3;
epsilon=10e-4;
mu=10e-6;
max_iterations=1000;
s_val = 20;



%% Generate response vector, data tensor, and initialize unknown parameter tensor

% generating response vector
y = [responses_arr_3];

% Labels 1,2,and 3 get mapped to 1
y(y~=0) = 1;
% Sample size
m = length(y);

% generating data tensor
A=zeros(m,n1,n2,n3);
i=1;

for j=1:length(responses_arr_3)
        A(i,:,:,:)=data_arr_3(j,:,:,:);
        i=i+1;
end

% clear loaded data
clear data_arr_3
clear responses_arr_3 

% Equivalent matrix
Amatrix=zeros(m,n1*n2*n3);
for i = 1:m
    Amatrix(i,:) = reshape(double(A(i,:,:,:)),n1*n2*n3,1);
end

%Normalization
for j=1:n1*n2*n3
    if norm(norm(Amatrix(:,j))) ~= 0
        Amatrix(:,j) = Amatrix(:,j) - mean(Amatrix(:,j));
    end
end

wr_tensor =[];
for exps = 1:experiments
    exps
    %% Recover the tensor W from A and y using Tensor IHT
    w0= abs(normrnd(0,1,[n1*n2*n3,1]));

    fg=@(w) norm(Amatrix*w - y)^2;

    % TPGD
    r1=r;r2=r;r3=r;
    diff=2*epsilon;
    s1new = s_val; 
    s2new = s_val;
    s3new = s_val;

    i=1;
    error_vector_iterations=[];
    wr=w0;
    while diff>epsilon && i<max_iterations
        % Descent step
        prev = wr;
        wr = wr - mu * Amatrix' * (Amatrix*wr - y);

        %Projection step
        What = cp_als(tensor(reshape(wr,[n1,n2,n3])),r);
        wr=reshape(double(What),[n1*n2*n3,1]); 

        diff = norm( fg(wr) - fg(prev) )/norm(fg(prev));

        i=i+1;
    end
    i
    wr_tensor = [wr_tensor wr];
end
save('adhd_cp_nyu.mat','r','mu','epsilon','wr_tensor','max_iterations','experiments','s1new',...
's2new','s3new')

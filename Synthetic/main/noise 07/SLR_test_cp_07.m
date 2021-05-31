%% Load packages
clear all; clc;
addpath(genpath('../../../tensor_toolbox'))
addpath(genpath('../../../tensorlab_2016-03-28'))
addpath(genpath('../../../SparsePCA_V2_0'))
addpath(genpath('../../../modules'))

%% Parameters (generative model)
seed_no = 11;
rng(seed_no)
noise_sigma=0.7;
n1=50;n2=50;n3=30;
s1=6;s2=6;s3=4;
r=3;
r1=r;r2=r;r3=r;
a=0.5;
% sample size vector
m_vector = 100:200:1500;

%% Parameters (algorithm)
max_iterations=5000;%for iht algorithm

% CP
ralgo = 2;  %chosen in validation experiments
no_of_experiments=50;
epsilon=10e-10;
mu=10e-2;

tensor_iht_arr = [];
for exp = 1:no_of_experiments
    exp
%% Generate low rank, sparse parameter tensor W
% Generate U1
U1 = zeros(n1,r1);
for k=1:r1
    vec=randperm(n1);
    signs=(ones(s1,1)*-1).^(binornd(1,0.5,[s1 1]));
    U1(vec(1:s1),k) = signs.* (a + abs(normrnd(0,1,s1,1)));
    U1(:,k) = U1(:,k)/norm(U1(:,k));
    %w1=w1./norm(w1);
end

% Generate U2
U2 = zeros(n2,r2);
for k=1:r2
    vec=randperm(n2);
    signs=(ones(s2,1)*-1).^(binornd(1,0.5,[s2 1]));
    U2(vec(1:s2),k) = signs.* (a + abs(normrnd(0,1,s2,1)));
    U2(:,k) = U2(:,k)/norm(U2(:,k));
    %w1=w1./norm(w1);
end

% Generate U3
U3 = zeros(n3,r3);
for k=1:r3
    vec=randperm(n3);
    signs=(ones(s3,1)*-1).^(binornd(1,0.5,[s3 1]));
    U3(vec(1:s3),k) = signs.* (a + abs(normrnd(0,1,s3,1)));
    U3(:,k) = U3(:,k)/norm(U3(:,k));
    %w1=w1./norm(w1);
end

% Generate core tensor D
D = 2 + 25*tenrand([r1 r2 r3]);

% Generate tensor W
W = ttm(ttm(ttm(D,U1,1),U2,2),U3,3);
wvec=reshape(double(W),[n1*n2*n3,1]);

%% Sample complexity experiments
tensor_iht = [];
for m = m_vector
%% Generate measurements
A=ten_normrnd(0,1/sqrt(m),[m,n1,n2,n3]);%tensor

Amatrix=zeros(m,n1*n2*n3);%equivalent matrix
for i=1:m
    Amatrix(i,:) = reshape(double(A(i,:,:,:)),n1*n2*n3,1);
end


%% Take measurements
y = Amatrix*wvec + normrnd(0,noise_sigma,m,1);


%% Recover the tensor W from X and y using Tensor IHT
diff=2*epsilon;

% Initialization of unknown tensor 
w0=normrnd(0,1,[n1*n2*n3,1]);
wr=w0;

fg=@(w) norm(Amatrix*w - y)^2;

i=1;
error_vector_iterations=[];
while diff>epsilon && i<max_iterations
    % Descent step
    prev = wr;
    wr = wr - mu * Amatrix' * (Amatrix*wr - y);
    
    %Projection step
    What = cp_als(tensor(reshape(wr,[n1,n2,n3])),ralgo);
    wr=reshape(double(What),[n1*n2*n3,1]);

    diff = norm( fg(wr) - fg(prev) );
    
    i=i+1;

    error_vector_iterations = [error_vector_iterations ; norm(wr-wvec)/norm(wvec)];
end
tensor_iht=[tensor_iht;norm(wr-wvec)/norm(wvec)];

end
tensor_iht_arr = [tensor_iht_arr tensor_iht];
save('test_cp_07.mat','r','mu','max_iterations',...
'epsilon','m_vector','a','tensor_iht_arr',...
'ralgo', 'noise_sigma','no_of_experiments','seed_no')

end
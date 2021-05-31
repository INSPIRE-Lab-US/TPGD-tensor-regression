figure
%% m=1100
bins = 0:0.05:1.5; 
load('test_glrsp_01.mat')
subplot(1,3,1)
histogram(tensor_iht_arr(6,:), bins)
title('TPGD')
ylim([0,50])
%xlabel('MSE')
ylabel('Frequency')


load('test_rauhut_01.mat')
subplot(1,3,2)
histogram(tensor_iht_arr(6,:), bins)
title('PGD-Tucker')
%xlabel('MSE')
ylim([0,50])
ylabel('Frequency')


load('test_cp_01.mat')
%bins = 0:0.05:2.2; 
subplot(1,3,3)
histogram(tensor_iht_arr(6,:), bins)
title('PGD-CP')
ylim([0,50])
ylabel('Frequency')
xlabel('Normalized Estimation Error')

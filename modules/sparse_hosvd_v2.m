function[U,V,W,D,Xhat] = sparse_hosvd_v2(x,s1,s2,s3,r1,r2,r3)
%ns = size(x);
%n = ns(1); p = ns(2); q = ns(3);
[U] = sparsePCA(double(tenmat(x,1))',s1,r1,1,0);
[V] = sparsePCA(double(tenmat(x,2))',s2,r2,1,0);
[W] = sparsePCA(double(tenmat(x,3))',s3,r3,1,0);
D = ttm(ttm(ttm(x,U',1),V',2),W',3);
Xhat = ttm(ttm(ttm(D,U,1),V,2),W,3);
end





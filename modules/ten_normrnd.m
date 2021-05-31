function X = ten_normrnd(mu,sigma,varargin)

if nargin == 1
    sz = varargin{1};
else
    sz = cell2mat(varargin);
end

data = normrnd(mu,sigma,[sz 1 1]);
X = tensor(data,sz);
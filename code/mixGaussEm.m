%% mixGaussEm: Perform EM algorithm for fitting the Gaussian mixture model.
 %% Input:
 %% X: [d, n], data matrix
 %% K: [1, 1], number of components in standard EM algorithm
 %% Output:
 %% Mu: [d, K], Mu of a Gaussian mixture models generated by standard EM algorithm
 %% Sigma: [d, d, K], Sigma of a Gaussian mixture models generated by standard EM algorithm
 %% Pi: [1, K], Pi of a Gaussian mixture models generated by standard EM algorithm
 %% label: [1, n], data label indicating each data belongs to which components
 %% llh: loglikelihood in each iteration
function [Mu, Sigma, Pi, label, llh] = mixGaussEm(X, K)
	tol = 1e-6;
	maxiter = 500;
	llh = -inf(1, maxiter);
	n = size(X, 2);
	label = ceil(K * rand(1, n));
	R = full(sparse(1:n, label, 1, n, K, n));
	for iter = 2 : maxiter
		[~, label(1, :)] = max(R, [], 2);
		R = R(:, unique(label));
		[Mu, Sigma, Pi] = maximization(X, R);
		[R, llh(iter)] = expectation(X, Mu, Sigma, Pi);
		if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter)); break; end;
	end
	llh = llh(2 : iter);
end


%% maximization
 %% Input:
 %% X: [d, n]
 %% R: [n, K]
 %% Output:
 %% Mu: [d, K]
 %% Sigma: [d, d, K]
 %% Pi: [1, K]
function [Mu, Sigma, Pi] = maximization(X, R)
	[d, n] = size(X);
	K = size(R, 2);
	Nk = sum(R, 1);
	Pi = Nk / n;
	Mu = bsxfun(@times, X * R, 1 ./ Nk);
	Sigma = zeros(d, d, K);
	r = sqrt(R);
	for k = 1 : K
		Xo = bsxfun(@minus, X, Mu(:, k));
		Xo = bsxfun(@times, Xo, r(:, k)');
		Sigma(:, :, k) = Xo * Xo' / Nk(k) + eye(d) * (1e-6);
	end
end


%% expectation
 %% Input:
 %% X: [d, n]
 %% Mu: [d, K]
 %% Sigma: [d, d, K]
 %% Pi: [1, K]
 %% Output:
 %% R: [n, K]
 %% llh: [1, 1]
function [R, llh] = expectation(X, Mu, Sigma, Pi)
	n = size(X, 2);
	K = size(Mu, 2);
	R = zeros(n, K);
	for k = 1 : K
		% R(i,j) : Probability that ith data belongs to jth gaussian distribution
		R(:,k) = loggausspdf(X, Mu(:,k), Sigma(:,:,k));
	end
	R = bsxfun(@plus, R, log(Pi)); % add prior probability
	T = logsumexp(R, 2); % T(i): Probability that ith data belongs to this model
	llh = sum(T) / n;
	R = exp(bsxfun(@minus, R, T));
end


%% loggausspdf
 %% Input:
 %% X: [d, n]
 %% Mu: [d, 1]
 %% Sigma: [d, d, 1]
 %% Output:
 %% y: [n, 1]
function [y] = loggausspdf(X, Mu, Sigma)
	d = size(X, 1);
	X = bsxfun(@minus, X, Mu);
	[U, p] = chol(Sigma);
	if p ~= 0
		error('ERROR: Sigma is not PD.');
	end
	Q = U'\X;
	q = dot(Q,Q,1);
	c = d * log(2 * pi) + 2 * sum(log(diag(U)));
	y = -(c + q) / 2;
end


%% logsumexp: Compute log(sum(exp(X),dim)) while avoiding numerical underflow.
%% By default dim = 1 (columns).
%% Written by Mo Chen (sth4nth@gmail.com).
 %% Input:
 %% X: [d, n]
 %% dim: [1, 1]
 %% Output:
 %% s: [n, 1]
function s = logsumexp(X, dim)
    % Compute log(sum(exp(X),dim)) while avoiding numerical underflow.
    %   By default dim = 1 (columns).
    % Written by Mo Chen (sth4nth@gmail.com).
    if nargin == 1, 
        % Determine which dimension sum will use
        dim = find(size(X)~=1,1);
        if isempty(dim), dim = 1; end
    end
    
    % subtract the largest in each dim
    y = max(X,[],dim);
    s = y+log(sum(exp(bsxfun(@minus,X,y)),dim));   % TODO: use log1p
    i = isinf(y);
    if any(i(:))
        s(i) = y(i);
    end
end
%% extendMixGaussEm: function description
 %% Input:
 %% X: [d, N]
 %% prePi: [1, N]
 %% preSigma: [d, d, N]
 %% H: [N, M]
 %% Output:
 %% Mu: [d, M]
 %% Sigma: [d, d, M]
 %% Pi: [1, M]
function [Mu, Sigma, Pi, llh] = extendMixGaussEm(X, M, prePi, preSigma)
	tol = 1e-4;
	maxiter = 500;
	llh = -inf(1, maxiter);
	[Mu, Sigma, Pi] = initialization(X, M);
	for iter = 2 : maxiter
		H = extendExpectation(X, Mu, Sigma, Pi, prePi, preSigma);
		[Mu, Sigma, Pi] = extendMaximization(X, H, prePi, preSigma);
		llh(iter) = cal_LLH(X, Mu, Sigma, Pi);
        fprintf('llh: %d\n', llh(iter));
		if abs(llh(iter)-llh(iter-1)) < tol * abs(llh(iter)); break; end;
	end
	llh = llh(2 : iter);
end


function [Mu, Sigma, Pi] = initialization(X, M)
	[d, n] = size(X);
	
	Pi = ones(1, M) / M;

	rndp = randperm(n);
	if M > n
        index = randi([1,n], 1, M);
    else
        index = rndp(1 : M);
    end
    Mu = X(:, index);

    X_mean = sum(X, 2) ./ n;
    X_minus = bsxfun(@minus, X, X_mean);
    tmp = X_minus * X_minus';
    tmp_pd = toPD(tmp ./ n);
    
    Sigma = zeros(d, d, M);
    for k = 1 : M
        Sigma(:,:,k) = tmp_pd;
    end
end


%% extendExpectation: function description
 %% Input:
 %% X: [d, N]
 %% Mu: [d, M]
 %% Sigma: [d, d, M]
 %% Pi: [1, M]
 %% Output:
 %% h: [N, M]
 %% llh: [1]
function H = extendExpectation(X, Mu, Sigma, Pi, prePi, preSigma)
	[d, N] = size(X);
	M = size(Mu, 2);
	H = zeros(N, M);
    preSigmaCell = mat2cell(preSigma, d, d, ones(1, N));
	for m = 1 : M
        SigmaCell = mat2cell(repmat(Sigma(:,:,m),1,1,N),d,d,ones(1,N));
        tmp_1 = mvnpdf(X', Mu(:, m)', toPD(Sigma(:, :, m)));
        tmp_2 = cellfun(@(x,y) x\y, SigmaCell, preSigmaCell, 'UniformOutput', false);
        tmp_3 = exp(-0.5 * reshape(cellfun(@trace, tmp_2), [N, 1]));
        H(:, m) = Pi(m) * ((tmp_1 .* tmp_3) .^ (prePi') );
	end
	S = sum(H, 2);
	H = bsxfun(@rdivide, H, S);
	NaN_idx = find(isnan(H(:, 1)));
	H(NaN_idx, :) = ones(length(NaN_idx), M) / M;
end

%% extendMaximization: function description
 %% Input:
 %% X: [d, N]
 %% H: [N, M]
 %% Output:
 %% Mu: [d, M]
 %% Sigma: [d, d, M]
 %% Pi: [1, M]
function [Mu, Sigma, Pi] = extendMaximization(X, H, prePi, preSigma)
	[d, N] = size(X);
	M = size(H, 2);

	Pi = zeros(1, M);
	Mu = zeros(d, M);
	Sigma = zeros(d, d, M);

	Pi = sum(H, 1) / N;

    for m = 1 : M
    	w_tmp = H(:, m) .* prePi';
    	w_tmp = w_tmp / sum(w_tmp);
    	Mu(:, m) = X * w_tmp;
		
		X_shift = bsxfun(@minus, X, Mu(:, m));
		X_shift_rep= repmat(reshape(X_shift, [d,1,N]), 1, d);
		sig_tmp = bsxfun(@times, X_shift_rep, permute(X_shift_rep, [2,1,3])) + preSigma;
		sig_tmp = sum(bsxfun(@times, reshape(w_tmp, [1,1,N]), sig_tmp), 3);
		Sigma(:,:,m) = toPD(sig_tmp);
    end    
end

%% toPD: translate matrix to PD matrix
function [Sigma] = toPD(Sigma)
	[V, D] = eig(Sigma);
	for i = 1 : size(Sigma, 1)
		if D(i, i) <= 0
			D(i, i) = 1e-4;
		end
	end
	Sigma = V * D * V';
	Sigma = (Sigma + Sigma') / 2;
end


function llh = cal_LLH(X, Mu, Sigma, Pi)
	[~, N] = size(X);
	M = size(Mu, 2);
	tmp = zeros(N, M);
	for m = 1 : M
		tmp(:, m) = Pi(m) * mvnpdf(X', Mu(:, m)', Sigma(:, :, m));
	end
	tmp = log(sum(tmp, 2));
	llh = sum(tmp) / N;
end
%% predict: label test data
 %% Input:
 %% img: test image
 %% wordTxt: [d, N]
 %% Model: [3, wordCnt] cell, each column refers to GMM model for a word
 %%        including Pi, Mu and Sigma
 %% k: [1,1], top k words
 %% Output:
 %% pros: [1, k], log probability that the image belongs to each word
 %% labels: [1, k], predict labels
function [pros, labels] = predict(img, Model, k)
	stride = 6; len = 8; reduction = true;
	X = imageDivision(img, stride, len, reduction);
	wordCnt = size(Model, 2);
	llh = zeros(1, wordCnt);
	for cnt = 1 : wordCnt
		Pi = Model{1,cnt}; Mu = Model{2,cnt}; Sigma = Model{3,cnt};
		N = size(X, 2); M = size(Pi, 2); H = zeros(N, M); 
		if size(Pi, 2) ~= 0
			llh(cnt) = cal_LLH(X, Mu, Sigma, Pi);
        end
		[pros, labels] = sort(llh, 'descend');
    	pros = pros(1 : k);
    	labels = labels(1 : k);
	end
end


%% cal_LLH
 %% Input:
 %% X: [d, N]
 %% Mu: [d, M]
 %% Sigma: [d, d, M]
 %% Pi: [1, M]
 %% Output:
 %% llh: [1,1]
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
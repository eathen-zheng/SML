%% word2data: function description
 %% Input:
 %% filename: a txt filename for a word, each line refers to a training image
 %% Output: 
 %% preMu: [d, N]
 %% prePi: [1, N]
 %% preSigma: [d, d, N]
function [preMu, prePi, preSigma] = word2data(baseDir, filename)
	d = 63; K = 8; stride = 6; len = 8; reduction = true;
	imgPathList = textread([baseDir, filename], '%s');
	imgCnt = size(imgPathList, 1);
    preMu = zeros(d, imgCnt * K);
    prePi = zeros(1, imgCnt * K);
    preSigma = zeros(d, d, imgCnt * K);
	for i = 1 : imgCnt
        fprintf('processing %dth image \n', i);
		imgPath = char(imgPathList(i));
		img = imread([baseDir, 'data/', imgPath, '.jpeg']);
		[X] = imageDivision(img, stride, len, reduction);
		[Mu, Sigma, Pi, ~, ~] = mixGaussEm(X, K);
        preMu(:, (i-1)*K+1 : i*K) = Mu;
        prePi(:, (i-1)*K+1 : i*K) = Pi;
        preSigma(:, :, (i-1)*K+1 : i*K) = Sigma;
	end
end
baseDir = '../../';
wordTxt = 'train/words.txt';
testTxt = 'data/test.txt';
trainTxt = 'data/train.txt';
wordList = textread([baseDir, wordTxt], '%s');
testList = textread([baseDir, testTxt], '%s');

load '../model/FinalModel.mat'
load '../data/test.mat'
load '../data/train.mat'

Row = 4; Col = 5; k = 5;

testCnt = size(testList, 1);
testIdxList = randperm(testCnt);
testIdxList = testIdxList(1 : Row * Col);

for plotIdx = 1 : Row * Col
	
	testIdx = testIdxList(plotIdx);

	imgPath = char(testList(testIdx));
	imgPath = [baseDir, 'data/', imgPath, '.jpeg'];
	img = imread(imgPath);
    
	[PredPros, PredLabelsIdx] = predict(img, Model, k);

	trueLabelIdx = find(testLabel(testIdx, :) == 1);
	trueTitle = '';
	for i = 1 : size(trueLabelIdx,2)
		trueTitle = strcat(trueTitle, char(wordList(trueLabelIdx(i))), {32});
	end
	
	predTitle = '';
	for i = 1 : size(PredLabelsIdx,2)
		predTitle = strcat(predTitle, char(wordList(PredLabelsIdx(i))), {32});
		
	end

	subplot(Row, Col, plotIdx); imshow(imgPath);
	title({['true labels: ', trueTitle{1,1}]; ['predict labels: ', predTitle{1,1}]});
end
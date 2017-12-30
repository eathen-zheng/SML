baseDir = '/home/zyc/Projects/web/';
wordTxt = 'data/words.txt';
FinalModel = genModel(baseDir, wordTxt);
%save FinalModel.mat FInalModel
save '../model/FinalModel.mat' FinalModel
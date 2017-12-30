%% genModel: function description
function [Model] = genModel(baseDir, wordTxt)
    wordList = textread([baseDir, wordTxt], '%s');
    wordCnt = size(wordList, 1);
    M = 64; Model = cell(3, wordCnt);
    fprintf('wordCnt: %d \n', wordCnt);
    for i = 1 : wordCnt
        filename = char(wordList(i));
        fprintf('gen No.%d model ', i);
        fprintf(filename)
        fprintf('\n')
        [preMu, prePi, preSigma] = word2data(baseDir, ['train/', filename]);
        if size(prePi, 2) > 64
            [Mu, Sigma, Pi, llh] = extendMixGaussEm(preMu, M, prePi, preSigma);
            Model{1, i} = Pi; Model{2, i} = Mu; Model{3, i} = Sigma;
        end
        if mod(i,10) == 0
            save(['../model/Model', num2str(i), '.mat'], 'Model')
        end
    end
end
%% imageDivision: extracts features using DCT on each channel of 3-dimention-image
 %% Input:
 %% image: [H, W, C], data
 %% stride: DCT filter stride
 %% len: DCT filter size
 %% reduction: wether to reduction dimention
 %% Output:
 %% X: [d, n]
function [X] = imageDivision(image, stride, len, reduction)
	X = [];
	[H, W, C] = size(image);
	for h = len : stride : H
		for w = len : stride : W
			x = [];
			for c = 1 : C
				x_channel = dct2(image(h-len+1 : h, w-len+1 : w, c));
				if reduction == true
					idx = [1, 2, 3, 4, 5, 6, ...
						len+1, len+2, len+3, len+4, len+5, ...
						2*len+1, 2*len+2, 2*len+3, 2*len+4, ...
						3*len+1, 3*len+2, 3*len+3, ...
						4*len+1, 4*len+2, 5*len+1];
					x_channel = reshape(x_channel(idx), [], 1);
				else
					x_channel = reshape(x_channel, [], 1);
				end
				x = [x; x_channel];
			end
			X = [X, x];
		end
	end
	X = normal(X);
end

function [Nor_X] = normal(X)
    n = size(X,2);
    XMax = repmat(max(X, [], 2), 1, n);
    XMin = repmat(min(X, [], 2), 1, n);
    Nor_X = (X - XMin) ./ (XMax - XMin);
end
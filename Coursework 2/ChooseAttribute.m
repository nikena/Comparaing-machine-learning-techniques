% 
% clear;
% load('facialPoints.mat');
% load('labels.mat');
% features = reshape(points, [150,132]);
% targets = labels;
% 

% This function aims to find the maximum gain from all possible attributes
% Output: 1. bestFeature = the attribute index corresponding maximum GAIN
%         2. bestThreshold =  the attribute value corresponding maximum GAIN
function [bestFeature, bestThreshold, bestEntropy] = ChooseAttribute(features, targets)
     
     [rows, cols] = size(features);
     bestGain = -inf;
     bestFeature = -inf;
     bestThreshold = -inf;
     bestEntropy = 0;
     for i = 1:cols  % iterate each feature
         for j = 1:rows % each feature's value as one possible threshold
             curThreshold = features(j ,i);
%              disp("i = "+i+", j = "+j);
             [gain, entropy] = getGain(features(:,i), targets, curThreshold);
             if gain >= bestGain
                 bestGain = gain;
                 bestFeature = i;
                 bestThreshold = curThreshold;
                 bestEntropy = entropy;
             end
         end
     end
     disp("best gain: "+bestGain);
     disp("bestFeature: "+bestFeature);
     disp("bestThreshold: "+bestThreshold);
end

function entropy = getEntropy(targets)
%------ General method --------
%     count = hist(targets,unique(targets));
%     entropy = 0;
%     for i = 1:length(count)
%         ratio = count(i)/totalNums;
%         entropy = entropy - ratio * log2(ratio);
%     end


%------- Binary method(efficient than general method) --------
    totalNums = length(targets);
    
    zerosRadio = length(find(targets == 0))/totalNums;
    onesRadio = length(find(targets == 1))/totalNums;
    entropy = 0;
    if zerosRadio ~= 0
        entropy = entropy - zerosRadio*log2(zerosRadio);
    end
    if onesRadio ~= 0
        entropy = entropy - onesRadio*log2(onesRadio);
    end


end

function [gain, entropy] = getGain(attribute, targets, threshold)
    entropy = getEntropy(targets);
    
    [sortAttribute, preIndex] = sort(attribute);
    sortTargets = targets(preIndex, :);
    thresholdIndex = find(sortAttribute == threshold);
    
    totalNums = size(targets, 1);
    lNums = thresholdIndex - 1;
    geNums = totalNums - lNums;
    
    lTargets = [];
    geTargets = [];
    if lNums ~= 0
        lTargets = sortTargets(1:lNums);
    end
    if geNums ~= 0
        geTargets = sortTargets(lNums+1:end);
    end   
        
    remainder = lNums/totalNums * getEntropy(lTargets) + geNums/totalNums * getEntropy(geTargets);   
    gain = entropy  - remainder;
end


% clear;
% load('facialPoints.mat');
% load('labels.mat');
% features = reshape(points, [150,132]);
% targets = labels;


% This function aims to find the maximum gain from all possible attributes
% Output: 1. bestFeature = the attribute index corresponding maximum GAIN
%         2. bestThreshold =  the attribute value corresponding maximum GAIN
function [bestFeature, bestThreshold] = ChooseAttribute(features, targets)
     
     [rows, cols] = size(features);
     bestGain = -inf;
     bestFeature = -inf;
     bestThreshold = -inf;
     
     for i = 1:cols  % iterate each feature
         for j = 1:rows % each feature's value as one possible threshold
             curThreshold = features(j ,i);
%              disp("i = "+i+", j = "+j);
             gain = getGain(features(:,i), targets, curThreshold);
             if gain >= bestGain
                 bestGain = gain;
                 bestFeature = i;
                 bestThreshold = curThreshold;
             end
         end
     end
     disp(bestFeature);
     disp(bestThreshold);
end

function entropy = getEntropy(targets)
    categoryOfTarget = categorical(targets); %tranfer double type to categorical type
    categoryCounts = countcats(categoryOfTarget); %obtain each category's count
    totalNum = sum(categoryCounts); %get the total number of target 
    
    entropy = 0;
    for i = 1:length(categoryCounts)
        posRetio = categoryCounts(i)/totalNum;
        entropy = entropy - posRetio * log2(posRetio);
    end
end

function gain = getGain(attribute, targets, threshold)
    entropy = getEntropy(targets);
     
    %if less than threshold, save current infor into lessT 
    %if greater or equal than threshold, save current infor into
    %greatOrEqual 
    lessT = [];
    greatOrEqualT = [];
    for i = 1:size(targets,1)
        if attribute(i) < threshold
%             rows = size(lessT, 1);
            lessT = [lessT; i attribute(i) targets(i)];
        else
%             rows = size(greatOrEqualT, 1);
            greatOrEqualT = [greatOrEqualT; i attribute(i) targets(i)];
        end
    end
    
    [labelRows, ~] = size(targets);
        
    LTRows = size(lessT,1); 
    GERows = size(greatOrEqualT,1);
       
    remainder = 0;
    if LTRows ~= 0
        remainder = remainder + LTRows/labelRows * getEntropy(lessT(:,3));
    end
    if GERows ~= 0
        remainder = remainder + GERows/labelRows * getEntropy(greatOrEqualT(:,3));
    end
%     remainder = LTRows/labelRows * getEntropy(lessT(:,3)) + GERows/labelRows * getEntropy(greatOrEqualT(:,3));   
    gain = entropy  - remainder;
  
end
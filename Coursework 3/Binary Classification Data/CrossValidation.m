%CrossValidation: needs altering to include SVM not just ANN.
function [ accuracy, confusionMatrices ] = CrossValidation( trainIdxGroups, testIdxGroups, input, target, kernel )
%CROSSVALIDATION Summary of this function goes here
%   Detailed explanation goes here

for i = 1:k
        trainIdx = trainIdxGroups(:,i); 
        testIdx = testIdxGroups(:,i);

        % Get train and test data from current fold
        curTrainLabels = input(:,trainIdx);
        curTrainTarget = target(:, trainIdx);
        curTestLabels = input(:, testIdx);
        curTestTarget = target(:, testIdx);

        % Train network
        [NET, TR] = train(NET, curTrainLabels, curTrainTarget);

        output = sim(NET, curTestLabels); %predict the test labels and return the outcomes  
        [Y_col, Ind_row] = max(output); % find the max one in each multiclass vector
        predict_output = zeros(classes, testSize(i)); % initialize predict output

        % Change output like this [0,0,0,0,1,0] 
        for j = 1 : testSize(i)
            predict_output(Ind_row(j), j) = 1;
        end
        
        % Get confusion matrix and from it recall and precision
        confusionMatrices(i,:,:) = confusionMatrix(curTestTarget, predict_output);
        recall(i,:) = 1:classes;
        precision(i,:) = 1:classes;
        sumTP = 0;
        for j = 1:classes
            recall(i,j) = confusionMatrices(i,j,j)/sum(confusionMatrices(i,j,:));
            precision(i,j) = confusionMatrices(i,j,j)/sum(confusionMatrices(i,:,j));
            sumTP = confusionMatrices(i,j,j) + sumTP;
        end
        accuracy(i) = sumTP / length(curTestTarget);
end
end
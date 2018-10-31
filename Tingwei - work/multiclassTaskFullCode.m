clear
load("emotions_data.mat");

num = length(y);
classes = max(y); % Get number of classes
k = 10;

target = zeros(num, classes);
for i = 1:num
    target(i, y(i)) = 1;
end

% Transpose data
input = x';    
target = target';

% Before performing 10-fold validation we need to randomise the data
RandIndex = randperm(length(input));
input = input(:, RandIndex);
target = target(:, RandIndex);

% split data into k fold and divide into train set and test set 
[trainIdxGroups, testIdxGroups, trainSize, testSize] = kFoldSplitData(num, k);


% This part used for tuning parameters
%======================================
iteration = 1;
f1scoresGroup = zeros(iteration,6);
accuracyGroup = zeros(iteration,1);
lrGroups = [0.1, 0.01, 0.001, 0.0001, 0.00001];
trainFunGroup = ["trainlm", "traingd", "traingda", "traingdm"];
failsGroup = [6,8,10,15,20];
goalGroup = [0.1, 0.01, 0.001, 0.0001, 0.00001];
epochGroup = [100, 300, 500, 700, 900];
regGroup = [0.1, 0.05, 0.01, 0.0001];
%======================================
for tuning = 1:iteration
    fprintf("Tuning times: %d\n", tuning);
    % Create emotions network
    trainingFcn = 'trainlm'; %traingd, traingda, traingdm, traingdx
    learningFcn = 'learngdm'; %learngd or learngdm
    NET = newff(input, target, 12, {'logsig','purelin'}, trainingFcn, learningFcn);

    % Modify train parameters
    NET.trainParam.show = 5;
    NET.trainParam.epochs = 100;
    NET.trainParam.goal = 1e-5;
    NET.trainParam.lr = 0.01;
    NET.trainParam.max_fail = 10; % validation check times (default is 6)   
    NET.performParam.regularization = 0.005;   % Adds a regularization value which will penalize large weights and help avoid overfitting 

    %Initialize recall, precison and confusion matrix
    recall = zeros(k, classes);
    precision = recall;
    confusionMatrices = zeros(k, classes, classes); % Index as (i,:,:) for individual matrix
    accuracy = zeros(1, k);

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

    % Calculate F1 scores
    fscores = zeros(k, classes); % Get every f score from precision and recall
    for i = 1:k
        for j = 1:classes
            fscores(i,j) = fscore(1, recall(i,j), precision(i,j));
        end
    end

    % Averages
    avgf1scores = 1:classes;
    for i = 1:classes
        avgf1scores(i) = mean(fscores(:,i));
    end
    f1scoresGroup(tuning, :, :) = avgf1scores;
    
    % Accuracy
    avgAccuracy = mean(accuracy);
    accuracyGroup(tuning) = avgAccuracy;
end



function fs = fscore(beta, recall, precision)
    fs = (1+(beta*beta))*((precision*recall)/((beta*beta*precision)+recall));
end

% Targets = Actual value, Outputs = Predicted value
% Returns confusion matrix where x is the predicted result and y is the
% actual result. cm(n,:) will give all results where 1 was the output class
% and cm(:,n) will give all results where 1 was the target class, in both
% of these the nth index is the number of true positives
function cm = confusionMatrix(targets, outputs)
    l = size(targets, 1);
    cm = zeros(l, l);

    for i = 1:length(targets)
        [~,x] = max(targets(:, i));
        [~,y] = max(outputs(:, i));
        cm(x,y) = cm(x,y) + 1;
    end
end





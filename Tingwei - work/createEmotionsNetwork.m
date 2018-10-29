clear
load("emotions_data.mat");

num = length(y); 
target = zeros(num, 6);
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

% Initialize performance array and accuracy array
performance = zeros(10, 1);
accuracy = zeros(10, 1);

% Ten-fold cross validation 
CVO = cvpartition(num,'KFold',10);
for i = 1:10
    trainIdx = CVO.training(i); 
    testIdx = CVO.test(i);
    
    % Get train and test data from current fold
    curTrainLabels = input(:,trainIdx);
    curTrainTarget = target(:, trainIdx);
    curTestLabels = input(:, testIdx);
    curTestTarget = target(:, testIdx);
    
    % Create emotions network
    trainingFcn = 'trainlm'; %traingd, traingda, traingdm, traingdx
    learningFcn = 'learngdm'; %learngd or learngdm
    NET = newff(curTrainLabels, curTrainTarget, [10 8], {'tansig','purelin'}, trainingFcn, learningFcn);

    % Modify train parameters
    NET.trainParam.show = 5;
    NET.trainParam.epochs = 100;
    NET.trainParam.goal = 1e-5;
    NET.trainParam.lr = 1e-3;
    NET.trainParam.max_fail = 8; % validation check times (default is 6)   
    NET.performParam.regularization = 0.1;   % Adds a regularization value which will penalize large weights and help avoid overfitting 
    % Train network
    [NET, TR] = train(NET, curTrainLabels, curTrainTarget);

    output = sim(NET, curTestLabels); %predict the test labels and return the outcomes  
    [Y_col, Ind_row] = max(output); % find the max one in each multiclass vector
    predict_output = zeros(6, CVO.TestSize(i)); % initialize predict output

    % Change output like this [0,0,0,0,1,0] 
    for j = 1 : CVO.TestSize(i)
        predict_output(Ind_row(j), j) = 1;
    end
    
    % Get confusion matrix c: confusion cm: confusion matrix
    [c, cm] = confusion(curTestTarget, predict_output);
    %create confusion matrix plot
    %plotconfusion(curTestTarget, predict_output);  
    
    performance(i) = perform(NET, curTestTarget, predict_output);
    accuracy(i) = 1 - c;
end

avgPerformance = mean(performance);
avgAccuracy = mean(accuracy);




function [kFoldRecall, kFoldPrecision, kFoldF1score, kFoldAccuracy] = kfold(points, labels, k) 

    [trainIdxGroups, testIdxGroups, trainSize, testSize] = KFoldSplitData(size(points, 1), k);
    
    kFoldPredicted = zeros(k, 15);
    kFoldActual = zeros(k, 15);
    
    %Initialize recall, precison and accuracy and F1 score
    kFoldAccuracy = zeros(k, 1);
    kFoldRecall = zeros(k, 1);
    kFoldPrecision = zeros(k, 1);
    kFoldF1score = zeros(k,1);
    
    for i = 1:k
    
        trainIdx = trainIdxGroups(:, i); 
        testIdx = testIdxGroups(:, i);
        
        trainX = points(trainIdx, :);
        trainY = labels(trainIdx);
        testX = points(testIdx, :);
        testY = labels(testIdx);
      
        kFoldActual(i,:) = testY;
        % call decisiontree on 90% of the data to build the tree
        tree = DecisionTreeLearning(trainX,  trainY);
        % the input sample and output the predicted outcome of the tree
        kFoldPredicted(i,:) = EvaluateTree(tree, testX);
        % calculate the performance
        [accuracy, recall, precision, f1score] = calPerformence(kFoldPredicted(i,:), testY);
        kFoldAccuracy(i) = accuracy;
        kFoldRecall(i) = recall;
        kFoldPrecision(i) = precision;
        kFoldF1score(i) = f1score;
    end
end


function fs = fscore(beta, recall, precision)
    fs = (1+(beta*beta))*((precision*recall)/((beta*beta*precision)+recall));
end

function outputs  = EvaluateTree(tree, inputs)
    num = size(inputs,1);
    outputs = zeros(1, num);
    %a temp tree to go through it and check the label
    for n = 1:num
        temptree = tree;

        while ~isempty(temptree.kids)
            if inputs(n, temptree.attribute) < temptree.threshold
                temptree = temptree.kids{1};
            else
                temptree = temptree.kids{2};
            end
        end
    outputs(n) = temptree.class;
    end
end

function [accuracy, recall, precision, f1score] = calPerformence(predicted, actual)
    TP = 0;
    FP = 0;
    TN = 0;
    FN = 0;
    num = length(predicted);
    for j = 1:num
        if actual(j) == predicted(j)
            if actual(j) == 1
                TP = TP + 1;
            else
                TN = TN + 1;
            end
        elseif predicted(j) == 1
            FP = FP + 1;
        else
            FN = FN + 1;              
        end            
    end
    accuracy = (TP + TN) / num;
    recall = TP/(TP + FN);
    precision = TP/(TP + FP);
    f1score = fscore(1, recall, precision);
end



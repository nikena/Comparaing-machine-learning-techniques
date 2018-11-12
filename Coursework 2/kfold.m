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
        %call decisiontree on 90% of the data to build the tree
        tree = DecisionTreeLearning(trainX,  trainY);
        
        %a temp tree to go through it and check the label
        for n = 1:testSize(i)

            temptree = tree;

            while ~isempty(temptree.kids)
                if testX(n, temptree.attribute) < temptree.threshold
                    temptree = temptree.kids{1};
                else
                    temptree = temptree.kids{2};
                end
            end
        kFoldPredicted(i,n) = temptree.class;
        end
        
%         plotconfusion(kFoldActual(i,:), kFoldPredicted(i,:));
        TP = 0;
        FP = 0;
        TN = 0;
        FN = 0;
        for j = 1:testSize(i)
            if kFoldActual(i,j) == kFoldPredicted(i,j)
                if kFoldActual(i,j) == 1
                    TP = TP + 1;
                else
                    TN = TN + 1;
                end
            elseif kFoldPredicted(i,j) == 1
                FP = FP + 1;
            else
                FN = FN + 1;              
            end            
        end
        kFoldAccuracy(i) = (TP + TN) / testSize(i);
        kFoldRecall(i) = TP/(TP + FN);
        kFoldPrecision(i) = TP/(TP + FP);
        kFoldF1score(i) = fscore(1, kFoldRecall(i), kFoldPrecision(i));
    end
end

function fs = fscore(beta, recall, precision)
    fs = (1+(beta*beta))*((precision*recall)/((beta*beta*precision)+recall));
end




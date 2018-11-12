function kfolddecisiontree = kfold(points, labels) 

    k = 10;
    
    %splitting data into 10 folds
    [trainIdxGroups, testIdxGroups, trainSize, testSize] = KFoldSplitData(size(points, 2), k);
    
            
    predicted = [];
    
    for i = 1:k
    
        trainIdx = trainIdxGroups(:, i); 
        testIdx = testIdxGroups(:, i);
        
        %training data 90%
        trainX = points(:, trainIdx);
        trainY = labels(trainIdx);
        
        %test data 10%
        testX = points(:, testIdx);
        testY = labels(testIdx);
       
        %call decisiontree on 90% of the data to build the tree
        tree = DecisionTreeLearning(trainX,  trainY);
        %DrawDecisionTree(tree);
        
        %a temporary tree needed to go through it and check the label
        %classifier
        for n = 1:size(testX, 2)

            temptree = tree;
            %if no kids then leaf node
            while ~isempty(temptree.kids)
              
                if testX(temptree.attribute, n) < temptree.threshold
                    temptree = temptree.kids{1};

                else
                    temptree = temptree.kids{2};
                end

            end
        
        predicted = [predicted temptree.class];
        kfolddecisiontree = f1score(predicted, testY);
        end
        
    end
end

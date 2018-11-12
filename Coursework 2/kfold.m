function kfolddecisiontree = kfold(points, labels) 

    k = 10;
    [trainIdxGroups, testIdxGroups, trainSize, testSize] = KFoldSplitData(size(points, 1), k);
    
            
    predicted = [];
    actual = [];
    
    for i = 1:k
    
        trainIdx = trainIdxGroups(:, i); 
        testIdx = testIdxGroups(:, i);
        
        trainX = points(trainIdx, :);
        trainY = labels(trainIdx);
        testX = points(testIdx, :);
        testY = labels(testIdx, :);
        
        testY = testY';
        actual = [actual testY];
    
        %call decisiontree on 90% of the data to build the tree
        tree = DecisionTreeLearning(trainX,  trainY);
    
        
        %a temp tree to go through it and check the label
    
        for n = 1:size(testX, 1)
            temptree = tree;

            while ~isempty(temptree.kids)
                if testX(n, temptree.attribute) < temptree.threshold
                    temptree = temptree.kids{1};

                else
                    temptree = temptree.kids{2};
                end
            end
        
        predicted = [predicted temptree.class];
        end
    end
    kfolddecisiontree = [predicted; actual];
end

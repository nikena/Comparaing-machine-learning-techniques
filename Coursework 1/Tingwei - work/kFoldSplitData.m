% @ Parameter  dataNums: the total number of data
% @ Parameter  k: the number of fold
% @ Return: train groups and test groups in binary number
function  [splitTrainGroups, splitTestGroups, trainSize, testSize] = kFoldSplitData(dataNums, k)
    % Initialize train group and test group with k columns
    splitTrainGroups = ones(dataNums, k);  
    splitTestGroups = zeros(dataNums, k);
    
    % Random data order in order to assign group
    perm = randperm(dataNums);
    
    % Assign number to each fold, and makse sure the maxNum - minNum <= 1
    oneTestNum = floor(dataNums / k); 
    testSize = zeros(1, k);
    trainSize = zeros(1, k);
    for i = 1:k
        testSize(i) = oneTestNum;
        trainSize(i) = dataNums - oneTestNum;
    end
    if k * oneTestNum ~= dataNums 
        remainder = dataNums - k*oneTestNum;
        numPlusGroup = randperm(k, remainder);   
        for i = 1:length(numPlusGroup)
            testSize(numPlusGroup(i)) = testSize(numPlusGroup(i)) + 1;
            trainSize(numPlusGroup(i)) = trainSize(numPlusGroup(i)) - 1;
        end
    end
    
    % Assign test group and train group
    cur = 0;
    for i = 1:k
        for j = 1:testSize(i)
            splitTestGroups(perm(j+cur),i) = 1;
        end
        cur = cur + testSize(i);
    end
    splitTrainGroups = splitTrainGroups - splitTestGroups;
    splitTestGroups = logical(splitTestGroups);
    splitTrainGroups = logical(splitTrainGroups);
end
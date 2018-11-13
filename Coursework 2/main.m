    clear
    load('labels');
    load('facialPoints');
    points = reshape(points, [132, 150])';
    tree = DecisionTreeLearning(points,  labels);
    DrawDecisionTree(tree);
    
    k = 10;
    [kFoldRecall, kFoldPrecision, kFoldF1score, kFoldAccuracy] = kfold(points, labels, k);
    load('labels');
    load('facialPoints');
    points = reshape(points, [132, 150]);
    tree = DecisionTreeLearning(points(1:119, :), labels);
    DrawDecisionTree(tree);
   
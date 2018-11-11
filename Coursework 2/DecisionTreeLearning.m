function decisionTree = DecisionTreeLearning(features, labels)

    decisionTree = struct();
    decisionTree.op = "";

    if all(labels == labels(1))
        decisionTree.kids = cell(0);
        decisionTree.attribute = 0;
        decisionTree.threshold = 0;
        decisionTree.class = labels(1);
        return
    else 
        [bestFeature, bestThreshold] = ChooseAttribute(features, labels);
    
        decisionTree.kids = cell(1:2);
        decisionTree.attribute = bestFeature;
        decisionTree.threshold = bestThreshold;
        decisionTree.op = num2str(bestFeature);
    
        leftData = features(features(:, bestFeature) < bestThreshold, :);
        rightData = features(features(:, bestFeature) >= bestThreshold, :);
    
        leftLabels = labels(features(:, bestFeature) < bestThreshold);
        rightLabels = labels(features(:, bestFeature) >= bestThreshold);
    
        if isempty(leftData)
            decisionTree.class = MajorityValue(leftLabels);
        else
            decisionTree.kids{1} = DecisionTreeLearning(leftData, leftLabels);
        end
        if isempty(rightData) 
            decisionTree.class = MajorityValue(leftLabels);
        else
            decisionTree.kids{2} = DecisionTreeLearning(rightData, rightLabels);
        end
    end
end
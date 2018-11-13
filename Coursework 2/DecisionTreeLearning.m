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
        [bestFeature, bestThreshold, entropy] = ChooseAttribute(features, labels);
        
        decisionTree.kids = cell(1:2);
        decisionTree.attribute = bestFeature;
        decisionTree.threshold = bestThreshold;
        decisionTree.op = "feature("+num2str(bestFeature)+") <"+num2str(bestThreshold)+"  entropy: "+ num2str(entropy);
    
        leftData = features(features(:, bestFeature) < bestThreshold, :);
        rightData = features(features(:, bestFeature) >= bestThreshold, :);
    
        leftLabels = labels(features(:, bestFeature) < bestThreshold);
        rightLabels = labels(features(:, bestFeature) >= bestThreshold);
    
        if isempty(leftData)
            decisionTree.class = MajorityValue(labels); % leftLables
        else
            decisionTree.kids{1} = DecisionTreeLearning(leftData, leftLabels);
        end
        if isempty(rightData) 
            decisionTree.class = MajorityValue(labels); % rightLables
        else
            decisionTree.kids{2} = DecisionTreeLearning(rightData, rightLabels);
        end
    end
end
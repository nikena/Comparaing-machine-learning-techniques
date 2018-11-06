% function DECISION-TREE-LEARNING(features, labels) returns a decision tree
% inputs: features, set of training example features
% labels, target labels for the training examples

function decisionTree = DecisionTreeLearning(features, labels)

    decisionTree = struct();
    x = 0;
 % if all examples have the same label then return a leaf node with label = the label   
    decisionTree.op = "";
    same = true;
    firstLabel = labels(1);
    for i=2:length(labels)
        if labels(i) == firstLabel
            same = true;
        else
            same = false;
            i = 150;
        end
    end
    if same == true
        decisionTree.kids = cell(0);
        decisionTree.attribute = 0;
        decisionTree.threshold = 0;
        decisionTree.class = firstLabel;
        return
    else
        % else
        % [best_feature, best_threshold] ß CHOOSE-ATTRIBUTE(features,targets)
        [bestFeature, bestThreshold] = ChooseAttribute(features, labels);
        
        % tree ß a new decision tree with root decision attribute best
        decisionTree.kids = cell(1:2);
        decisionTree.attribute = 0;
        decisionTree.threshold = 0;
        decisionTree.class = bestFeature;
        
        % for each value υi of best do

        for j=1:length(bestFeature)
            
            % add a branch to tree corresponding to best = υi
            % {examplesi , targetsi}ß {elements of examples with best = υi and
            % corresponding targets}
            [subFeatures, subLabels] = ;
            
%             decisionTree.kids = cell{x + 1};
%             x = x+1;

            % if examplesi is empty then return a leaf node with label =
            % MAJORITY- VALUE(targets)
            % else subtree ß DECISION-TREE-LEARNING(examplesi , targets)
            % return tree
            if subFeatures == []
                decisionTree.kids{j} = MajorityValue(labels);
                return
            else
                decisionTree.kids{j} = DecisionTreeLearning(subFeatures, subLabels);
            end
        end  
    end
end
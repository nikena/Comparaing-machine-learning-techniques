function [binarySmileNetwork ] = BinarySmile()
    %load in and reshape data so it can be trained.
    load('facialPoints.mat');
    points = reshape(points, [132, 150]);
    load('labels.mat');
    %Invert labels in order to match the points (input) data.
    labels = labels';

    %Create a new binarySmileNetwork with 10 hidden nodes.
    binarySmileNetwork = newff(points, labels, 10);

    binarySmileNetwork.name = 'BinarySmile';
    binarySmileNetwork.trainParam.epochs = 100;
    binarySmileNetwork.trainParam.show = 7;
    binarySmileNetwork.trainParam.goal = 0;
    
%     k = 10;
%     binarySmileNetwork = train(binarySmileNetwork, points, labels);

    % Training setup
    k = 10; % k in k-cross validation
    s = size(points, 2); % size of input set
    testInd = round(s*0.9, 0):s; % Last 10% is test set
    binarySmileNetwork.divideParam.testInd = testInd;
    s = round(s*0.9, 0) - 1; % Decrease size of set as we don't want to use test set in training / validation
    
    % Training loop
    for i = 1:k
        validInd = ((i-1)*(round(s/k,0)))+1:i*(round(s/k,0)); % Indicies in validation set, all else in train set
        binarySmileNetwork.divideParam.valInd = validInd;
        binarySmileNetwork.divideParam.trainInd = setdiff(1:s, validInd); % Larger set on left of setdiff
        binarySmileNetwork = train(binarySmileNetwork, points, labels);
    end
end
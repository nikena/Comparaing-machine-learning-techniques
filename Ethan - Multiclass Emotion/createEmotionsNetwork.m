function emotionsNetwork = createEmotionsNetwork()
    % Load emotions data, must be in same folder
    load("emotions_data.mat");
    
    % Transpose x and y
    x=x';
    ySize = length(y);
    target = zeros(ySize, 6);
    for i = 1:ySize
        target(i, y(i)) = 1;
    end
    y=target';
    
    % Create emotions network
    emotionsNetwork = newff(x, y, 10);
    
    % Modify train parameters
    emotionsNetwork.trainParam.show = 5;
    emotionsNetwork.trainParam.epochs = 100;
    
    % Training setup
    k = 10; % k in k-cross validation
    s = size(x, 2); % size of input set
    
    indices = cvpartition(ySize,'KFold', k);
        
    %valInd = round(s*0.9, 0):s; % Last 10% is validation set
    %emotionsNetwork.divideParam.valInd = valInd;
    %s = round(s*0.9, 0) - 1; % Decrease size of set as we don't want to use validation set in training / testing
    
    accuracy = 1:k;
    
    % Training loop
    for i = 1:k
        % Assign test and training indicies
        testLabels = x(:, indices.test(i));
        testTarget = y(:, indices.test(i));
        trainLabels = x(:, indices.training(i));
        trainTarget = y(:, indices.training(i));
        
        % Train
        emotionsNetwork = train(emotionsNetwork, trainLabels, trainTarget);
        
        % Get output from test
        out = sim(emotionsNetwork, testLabels);
        [unused, index] = max(out);
    
        % Convert output to classification
        outMax = zeros(6, indices.TestSize(i));
        for j = 1 : indices.TestSize(i)
            outMax(index(j), j) = 1;
        end
        
        % Confusion matrix
        confusionMatrix = myConfusion(testTarget, outMax);
        
        accuracy(i) = 1-c;
    end
    
    plot(accuracy, '-x', 'MarkerIndices', 1:k)
    title("Accuracy of network on each fold");
    xlabel("k");
    ylabel("Accuracy (Mean = "+mean(accuracy)+")");
    axis([1, k, 0, 1]);
end

function confusionMatrix = myConfusion(targets, outputs)
    l = size(targets, 1);
    confusionMatrix = zeros(l, l);
    for i = 1:length(targets)
        [a,x] = max(targets(:, i));
        [a,y] = max(outputs(:, i));
        confusionMatrix(x,y) = confusionMatrix(x,y) + 1;
    end
end
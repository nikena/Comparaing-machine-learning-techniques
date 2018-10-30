function emotionsNetwork = createEmotionsNetwork()
    % Load emotions data, must be in same folder
    load("emotions_data.mat");
    
    % Transpose x and y
    x=x';
    y=y';
    
    % Create emotions network
    emotionsNetwork = newff(x, y, 10);
    
    % Divide by indexes
    emotionsNetwork.divideFcn = 'divideind';
    
    % Modify train parameters
    emotionsNetwork.trainParam.show = 5;
    emotionsNetwork.trainParam.epochs = 100;
    
    % Training setup
    k = 10; % k in k-cross validation
    s = size(x, 2); % size of input set
    valInd = round(s*0.9, 0):s; % Last 10% is validation set
    emotionsNetwork.divideParam.valInd = valInd;
    s = round(s*0.9, 0) - 1; % Decrease size of set as we don't want to use validation set in training / testing
    
    % Training loop
    for i = 1:k
        testInd = ((i-1)*(round(s/k,0)))+1:i*(round(s/k,0)); % Indicies in test set, all else in training set
        emotionsNetwork.divideParam.testInd = testInd;
        emotionsNetwork.divideParam.trainInd = setdiff(1:s, testInd); % All non-testInd into training set Larger set on left of setdiff
        emotionsNetwork = train(emotionsNetwork, x, y);
    end
end
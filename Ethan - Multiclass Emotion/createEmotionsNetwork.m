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
    testInd = round(s*0.9, 0):s; % Last 10% is test set
    emotionsNetwork.divideParam.testInd = testInd;
    s = round(s*0.9, 0) - 1; % Decrease size of set as we don't want to use test set in training / validation
    
    % Training loop
    for i = 1:k
        validInd = ((i-1)*(round(s/k,0)))+1:i*(round(s/k,0)); % Indicies in validation set, all else in train set
        emotionsNetwork.divideParam.valInd = validInd;
        emotionsNetwork.divideParam.trainInd = setdiff(1:s, validInd); % Larger set on left of setdiff
        emotionsNetwork = train(emotionsNetwork, x, y);
    end
end
clear;
load('facialPoints.mat');
points = reshape(points, [132, 8955]);
% points = mapminmax(points);



load('headpose.mat');
% 6th label of every data point 
labels = pose(:,6);
labels = labels';


k = 10;


% split data into k fold and divide into train set and test set 

[trainIdxGroups, testIdxGroups, trainSize, testSize] = KFoldSplitData(length(points), k);




actual = [];
predicted = [];
    
for i = 1:k

    %trainlm, traingd - no, traingda, traingdm, traingdx-default
    %learngdm - default or learngd 

    
    net = newff(points, labels, 20, {'logsig' 'purelin'}, 'trainlm', 'learngdm');
    
    
    net.trainParam.show = 10;
    net.trainParam.epochs = 120;
    net.trainParam.goal=1e-9;
    net.trainParam.max_fail = 8; % validation check times (default is 6) 
    net.trainParam.lr = 0.001;


    
    trainIdx = trainIdxGroups(:,i); 
    testIdx = testIdxGroups(:,i);

        
    trainX = points(:, trainIdx);
    trainY = labels(trainIdx);
    testX = points(:, testIdx);
    testY = labels(testIdx);


    %train(net, inputs, taregts)
    net = train(net, trainX, trainY);
    t = sim(net, testX);
    predicted = [predicted t];
    actual = [actual testY];
end
save net;
rmse = rms(actual - predicted);

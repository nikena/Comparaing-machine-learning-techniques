
function RegressionSet = cw1Regression()
    load('facialPoints.mat');
    load('headpose.mat');
    labels = pose(:,6);
    points = reshape(points, 132, 8955);
    points = points';
%      labels = labels';
%      pose = pose';
    indices = crossvalind('Kfold', points(1:8955, 132), 10);
    indices = indices';
    points = points';
    labels = labels';
    actual = [];
    predicted = [];
    for k=1:10
        testPoint = (indices == k);
        trainPoint = ~testPoint;
        trainPoint_data = points(:, trainPoint);
        trainPoint_target = labels(:, trainPoint);
        testPoint_data = points(:,testPoint);
        testPoint_target = labels(:, testPoint);
        netWork = newff(trainPoint_data, trainPoint_target, 10);
        netWork.trainParam.show = 10;
        netWork.trainParam.epochs=100;
        netWork.trainParam.goal = 1e-6;
        netWork = train(netWork, trainPoint_data, trainPoint_target);
        output = sim(netWork, testPoint_data);
        predicted = [predicted output];
        actual = [actual testPoint_target];
        
    end
    rmse = rms(actual - predicted);
    plot(rmse);
    
end
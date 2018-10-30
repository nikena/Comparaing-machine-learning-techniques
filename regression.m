clear;
load('facialPoints.mat');
points = reshape(points, [132, 8955]);
points = mapminmax(points);



load('headpose.mat');
% 6th label of every data point 
labels = pose(:,6);
labels = labels';


k = 10;
cv = cvpartition(size(points, 2), 'kfold',k);
actual = [];
predicted = [];
    
for i = 1:cv.NumTestSets

    %trainlm, traingd, traingda, traingdm, traingdx
    %learngdm, learngd or learngdm

    
    net = newff(points, labels, 10, {'tansig' 'purelin'});
    
    
    net.trainParam.show = 10;
    net.trainParam.epochs = 100;
    %net.numLayers = 2;
    net.trainParam.goal=1e-6;
    %net.biasConnect(1) = 1;
    %net.biasConnect(2) = 1;
    %net.inputConnect(1,1) = 1;
    %net.inputConnect(2,1) = 1;

    
    trainX = points(:, cv.training(i));
    trainY = labels(cv.training(i));
    testX = points(:, cv.test(i));
    testY = labels(cv.test(i));


    %train(net, inputs, taregts)
    net = train(net, trainX, trainY);
    t = sim(net, testX);
    predicted = [predicted t];
    actual = [actual testY];
    
end

rmse = rms(actual - predicted);

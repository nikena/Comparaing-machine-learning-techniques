function RegressionSet = cw1Regression()
    load('facialPoints.mat');
    load('headpose.mat');
    labels = pose(:,6);
    points = reshape(points, 132, 8955);
    labels = labels';
    pose = pose';
    netWork1 = newff(points, labels, 10);
    newWork2 = newff(pose, labels, 10);
    [netWork1, TR1] = train(netWork1, points, labels);
    [netWork2, TR2] = train(netWork2, pose, labels);
    t1 = sim(netWork1, points);
    t2 = sim(netWork2, pose);
    
end
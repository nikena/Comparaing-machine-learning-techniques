load('facialPoints.mat');
load('labels');
points = reshape(points, [132, 150])';
labels = labels';

num = length(labels);
k = 10;
kernelOptions = {0.1,0.2,0.5,0.8,1.0};
polyScale = {0.2,0.5,1.0,1.5,2.0};

MdlBCLinear = fitcsvm(points,labels,'KernelFunction','linear', 'BoxConstraint', 1);

%innerfold cross validation to discover best kernel value. Grid Search.
for i = 1 : length(kernelOptions)
    for j = 1 : k
    [trainIdxGroups, testIdxGroups, trainSize, testSize] = kFoldSplitData(num, k);   
    trainIdx = trainIdxGroups(:,i); 
    testIdx = testIdxGroups(:,i);

    % Get train and test data from current fold
    curTrainLabels = input(:,trainIdx);
    curTrainTarget = target(:, trainIdx);
    curTestLabels = input(:, testIdx);
    curTestTarget = target(:, testIdx);
    
    MdlBCLinear = fitcsvm(currentTrainLabels,currentTrainTargets,'KernelFunction','linear', 'BoxConstraint', 1, 'kernelScale',kernelOptions(j));
    end
end
best = max(accuracy);
MdlBCGaussianRBF = fitcsvm(points,labels, 'KernelFunction','RBF', 'BoxConstraint',1, 'KernelScale',best);

MdlBCPolynomial = fitcsvm(points,labels,'KernelFunction','polynomial', 'BoxConstraint', 1, 'PolynomialOrder',polyScale);
%% Regression Performence
clear
load 'facialPoints.mat'
load 'headpose.mat'

X = reshape(points, 132, 8955)'; % after reshape and transpose => 8955 x 132 double
Y = pose(:,6); % data format: 8955 x 1 double
k = 10;
rmseList_l = zeros(1, k);
rmseList_RBF = zeros(1, k);
rmseList_p = zeros(1, k);
rmseList_ANN = zeros(1,k);
predictSVM = [];
predictANN = [];
[out_TrainGroups, out_TestGroups, out_TrainSize, out_TestSize] = KFoldSplitData(size(Y,1), 10);
for i = 1:k
    disp("k = "+i);
    out_trainIdx = out_TrainGroups(:,i); 
    out_testIdx = out_TestGroups(:,i);

    % get train data in outer loop cross validation    
    out_trainX = X(out_trainIdx, :);
    out_trainY = Y(out_trainIdx);
    % get test data in outer loop cross validation 
    out_testX = X(out_testIdx, :);
    out_testY = Y(out_testIdx);
    
    % Record different models's results
    
    % train the SVM-linear kernel, and save the performance
    SVM_L = fitrsvm(out_trainX, out_trainY,'KernelFunction',"linear", 'BoxConstraint', 1, 'Epsilon', 1);
    rmseList_l(i) = sqrt(loss(SVM_L, out_testX, out_testY));
    disp(rmseList_l(i));
    predictSVM = [predictSVM, predict(SVM_L, out_testX)'];
   
    
    % train the SVM-RBF kernel, and save the performance
    SVM_RBF = fitrsvm(out_trainX, out_trainY,'KernelFunction',"RBF", 'BoxConstraint', 100, 'Epsilon', 0.5, 'KernelScale', 50);
    rmseList_RBF(i) = sqrt(loss(SVM_RBF, out_testX, out_testY));
    disp(rmseList_RBF(i));
    predictSVM = [predictSVM, predict(SVM_RBF, out_testX)'];
    
    % train the SVM-polynomial kernel, and save the performance
    SVM_P = fitrsvm(out_trainX, out_trainY,'KernelFunction',"polynomial", 'BoxConstraint', 0.1, 'Epsilon', 1, 'PolynomialOrder', 1);
    rmseList_p(i) = sqrt(loss(SVM_P, out_testX, out_testY));
    disp(rmseList_p(i));
    predictSVM = [predictSVM, predict(SVM_P, out_testX)'];
    
    % train the ANN kernel, and save the performance
    NET_ANN = newff(out_trainX', out_trainY', 20, {'logsig' 'purelin'}, 'trainlm', 'learngdm');    
    NET_ANN.trainParam.show = 10;
    NET_ANN.trainParam.epochs = 120;
    NET_ANN.trainParam.goal=1e-9;
    NET_ANN.trainParam.max_fail = 15;
    NET_ANN.trainParam.lr = 0.001;
    NET_ANN = train(NET_ANN, out_testX', out_testY');
    t = sim(NET_ANN, out_testX');
    predicted = t;
    predictANN = [predictANN, t];  % we record all the predict label and use to ttest2
    actual = out_testY';
    rmseList_ANN(i) = rms(actual - predicted);
    disp(rmseList_ANN(i));
end


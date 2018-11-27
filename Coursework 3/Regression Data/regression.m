
clear;
load 'facialPoints.mat'
load 'headpose.mat'
X = reshape(points, 132, 8955)'; % after reshape and transpose => 8955 x 132 double
Y = pose(:,6); % data format: 8955 x 1 double

% hyper-parameter possible value
KernelFunction = ["linear", "RBF", "polynomial"];
svmtype = KernelFunction(3); % change the kernel function type here

C = [0.1, 1, 100]; 
Epsilon = [0.3, 0.5, 1.0]; 
Sigma = [1, 50, 100]; % use for RBF kernel scale"
Q = [1, 2, 3];% use for Polynomial order

% inner-fold-cross-validation use for finding optimal hyperparameters
n = 3;
% k-cross-validation
k = 10;

% initial tunning performance outcome matrix, using for record all the
% tuning outcome by grid searh
TuningPerfList = [];
paraNums = 0;
switch (svmtype)
    case "linear"
        TuningPerfList = zeros(length(C)*length(Epsilon),5,n); % save linear performance results
        paraNums = 2;
    case "RBF"
        TuningPerfList = zeros(length(C)*length(Epsilon)*length(Sigma),6,n); % save RBF tuning parameter results
        paraNums = 3;
    case "polynomial"
        TuningPerfList = zeros(length(C)*length(Epsilon)*length(Q),6,n); % save Polynomial parameter results
        paraNums = 3;
end
bestValue = zeros(1, paraNums+3); %initial best Parameter
rmseList = zeros(1, k); % initial rmse array using for save each rmse in each iteration (10-fold)
[out_TrainGroups, out_TestGroups, out_TrainSize, out_TestSize] = KFoldSplitData(size(Y,1), 10);
for i = 1:k
    out_trainIdx = out_TrainGroups(:,i); 
    out_testIdx = out_TestGroups(:,i);

    % get train data in outer loop cross validation    
    out_trainX = X(out_trainIdx, :);
    out_trainY = Y(out_trainIdx);
    % get test data in outer loop cross validation 
    out_testX = X(out_testIdx, :);
    out_testY = Y(out_testIdx);
    
    
    % beginning of inner loop, only for the first iteration of outer loop to
    % tune parameters
    
    if i == 1
        [in_TrainGroups, in_TestGroups, in_TrainSize, in_TestSize] = KFoldSplitData(out_TrainSize(i), n);                  
        for j = 1:n
            in_trainIdx = in_TrainGroups(:,j); 
            in_testIdx = in_TestGroups(:,j);
            
            % get train data in inner loop cross validation   
            in_trainX = out_trainX(in_trainIdx, :);
            in_trainY = out_trainY(in_trainIdx);
            % get test data in inner loop cross validation 
            in_testX = out_trainX(in_testIdx, :);
            in_testY = out_trainY(in_testIdx);
            
            % type of svm to work with if we want to tune rbf
            % 
            if svmtype == "RBF"
                num = 0;
                totalNum = length(C)*length(Epsilon)*length(Sigma);
                for c = 1:length(C)
                    for e = 1: length(Epsilon)
                        for s = 1: length(Sigma)
                            Mdl = fitrsvm(in_trainX, in_trainY,'KernelFunction',"RBF", 'BoxConstraint', C(c), 'Epsilon', Epsilon(e), 'KernelScale', Sigma(s));
                            mseVal = loss(Mdl, in_testX, in_testY);
                            rmse = sqrt(mseVal);
                            num = num + 1;                      
                            supportVNums = size(Mdl.SupportVectors,1);
                            supportVRatio = supportVNums/size(in_trainX,1);
                            TuningPerfList(num,:,j) = [C(c),Epsilon(e),Sigma(s),supportVNums, supportVRatio, rmse];
                            disp("tuning iteration: "+j+"/"+n+", KernelFunction: "+svmtype + ", times: "+num+"/"+totalNum+", supVec Nums: "+supportVNums+", supVec Ratio: "+supportVRatio+", rmse: "+ rmse);
                        end
                    end
                end
            elseif svmtype == "polynomial"
                num = 0;
                totalNum = length(C)*length(Epsilon)*length(Q);
                for c = 1:length(C)
                    for e = 1: length(Epsilon)
                        for q = 1: length(Q)
                            Mdl = fitrsvm(in_trainX, in_trainY,'KernelFunction',"polynomial", 'BoxConstraint', C(c), 'Epsilon', Epsilon(e), 'PolynomialOrder', Q(q));
                            mseVal = loss(Mdl, in_testX, in_testY);
                            rmse = sqrt(mseVal);
                            num = num + 1;
                            supportVNums = size(Mdl.SupportVectors,1);
                            supportVRatio = supportVNums/size(in_trainX,1);
                            TuningPerfList(num,:,j) = [C(c),Epsilon(e),Q(q),supportVNums, supportVRatio,rmse];
                            disp("tuning iteration: "+j+"/"+n+", KernelFunction: "+svmtype + ", times: "+num+"/"+totalNum+", supVec Nums: "+supportVNums+", supVec Ratio: "+supportVRatio+", rmse: "+ rmse);
                        end
                    end
                end
            else
                num = 0;
                totalNum = length(C)*length(Epsilon);
                for c = 1:length(C)
                    for e = 1: length(Epsilon)
                        Mdl = fitrsvm(in_trainX, in_trainY,'KernelFunction',"linear", 'BoxConstraint', C(c), 'Epsilon', Epsilon(e),'Standardize',true);
                        mseVal = loss(Mdl, in_testX, in_testY);
                        rmse = sqrt(mseVal);
                        num = num + 1;
                        supportVNums = size(Mdl.SupportVectors,1);
                        supportVRatio = supportVNums/size(in_trainX,1);
                        TuningPerfList(num,:,j) = [C(c),Epsilon(e),supportVNums, supportVRatio, rmse];
                        disp("tuning iteration: "+j+"/"+n+", KernelFunction: "+svmtype + ", times: "+num+"/"+totalNum+", supVec Nums: "+supportVNums+", supVec Ratio: "+supportVRatio+", rmse: "+ rmse);
                    end
                end
            end            
        end
        avg_tuningPerfList = (TuningPerfList(:,:,1) + TuningPerfList(:,:,2) + TuningPerfList(:,:,3)) / 3;
        [bestRmse, bestIndex] = min(avg_tuningPerfList(:,paraNums+3));
        bestValue = avg_tuningPerfList(bestIndex,:);
        disp(bestValue);
    end    
   
    % train the SVM model using best parameters which tunned by grid search
    switch(svmtype)
        case "linear"
            SVM = fitrsvm(out_trainX, out_trainY,'KernelFunction',"linear", 'BoxConstraint', bestValue(1), 'Epsilon', bestValue(2));
        case "RBF"
            SVM = fitrsvm(out_trainX, out_trainY,'KernelFunction',"RBF", 'BoxConstraint', bestValue(1), 'Epsilon', bestValue(2), 'KernelScale', bestValue(3));
        case "polynomial"
            SVM = fitrsvm(out_trainX, out_trainY,'KernelFunction',"polynomial", 'BoxConstraint', bestValue(1), 'Epsilon', bestValue(2), 'PolynomialOrder', bestValue(3));
    end
    mseVal = loss(SVM, out_testX, out_testY);
    rmseList(i) = sqrt(mseVal);
    disp("Train SVM iteration: "+i+"/" +k+ " ,rmse: "+rmseList(i));
end
avgRmseList =  mean(rmseList);
disp("Avg 10 Fold rmse: " + avgRmseList);

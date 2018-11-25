clear;
load 'facialPoints.mat'
load 'headpose.mat'
X = reshape(points, 132, 8955)';
Y = pose(:,6);

% hyper-parameter possible value
KernelFunction = ["RBF", "Polynomial"];
C = [1, 10, 20, 30, 40, 50, 60, 100, 1000, 10000];
Epsilon = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0];
Sigma = [0.001, 0.01, 0.1, 10, 50, 80, 90, 100, 1000, 10000]; % use for RBF"
Q = [1, 2, 3, 4, 5, 6]; % use for Polynomial


disp(KernelFunction(2));
% k-cross-validation
k = 10;
% inner-fold-cross-validation use for finding optimal hyperparameters
n = 3;

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
 
    [in_TrainGroups, in_TestGroups, in_TrainSize, in_TestSize] = KFoldSplitData(out_TrainSize(i), n);
    for j = 1:n
        in_trainIdx = in_TrainGroups(:,i); 
        in_testIdx = in_TestGroups(:,i);

        % get train data in inner loop cross validation   
        in_trainX = out_trainX(in_trainIdx, :);
        in_trainY = out_trainY(in_trainIdx);
        % get test data in inner loop cross validation 
        in_testX = out_trainX(in_trainIdx, :);
        in_testY = out_trainY(in_trainIdx);
        
        
        % ----when kernel method is RBF ----------
%         for s = 1:length(Sigma)
%             Mdl = fitrsvm(in_trainX, in_trainY,'KernelFunction',"RBF", 'BoxConstraint',1, 'Epsilon', 0.3, 'KernelScale', Sigma(s));
%             mseVal = loss(Mdl, in_testX, in_testY);
%             rmse = sqrt(mseVal);
%             disp(" RBF KernelScale: "+Sigma(s)+ ", rmse = "+rmse);
%         end
%         disp('\n');
%         
%         for c = 1:length(C)
%             Mdl = fitrsvm(in_trainX, in_trainY,'KernelFunction',"RBF", 'BoxConstraint',C(c), 'Epsilon', 0.3, 'KernelScale', 80);
%             mseVal = loss(Mdl, in_testX, in_testY);
%             rmse = sqrt(mseVal);
%             disp(" RBF BoxConstraint: "+C(c)+ ", rmse = "+rmse);
%         end
        
        for e = 1:length(Epsilon)
            Mdl = fitrsvm(in_trainX, in_trainY,'KernelFunction',"RBF", 'BoxConstraint',2, 'Epsilon', Epsilon(e), 'KernelScale', 80);
            mseVal = loss(Mdl, in_testX, in_testY);
            rmse = sqrt(mseVal);
            disp(" RBF Epsilon: "+Epsilon(e)+ ", rmse = "+rmse);
        end
        % ------------------end-------------------
        
        
        % ----when kernel method is Polynomial---- 
        for q = 1:length(Q)
            
        end 
        
        for c = 1:length(C)
            
        end
        
        for e = 1:length(Epsilon)
            
        end
        % -----------------end--------------------
        for kf = 1:length(KernelFunction)
            Mdl = fitrsvm(in_trainX, in_trainY,'KernelFunction',KernelFunction(kf), 'BoxConstraint',1, 'Epsilon', 0.3, 'KernelScale',40);
            mseVal = loss(Mdl, in_testX, in_testY);
            rnse = sqrt(mseVal);
            disp("kf: "+KernelFunction(kf) + "\n rnse = "+rnse + "\n");
        end
       
    end
    
end

% function Mdl = trainSVM(kFuntion, )
%     Mdl = fitrsvm(in_trainX, in_trainY,'KernelFunction',KernelFunction(kf), 'BoxConstraint',1, 'Epsilon', 0.3, 'KernelScale',40);
%     mseVal = loss(Mdl, in_testX, in_testY);
%     rnse = sqrt(mseVal);
%     disp("kf: "+KernelFunction(kf) + "\n rnse = "+rnse + "\n");
% end

% Mdl = fitrsvm(X,Y,'KernelFunction','Polynomial', 'KernelScale',100, 'BoxConstraint',0.1, 'Epsilon', 0.3);
% 
% 
% 
% mseVal = loss(Mdl, X, Y);
% rnse = sqrt(mseVal);
% avg_rmse(e) = sum(rmse) / k;
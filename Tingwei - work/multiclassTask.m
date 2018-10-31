clear
load("emotions_data.mat");

num = length(y);
classes = max(y); % Get number of classes

target = zeros(num, classes);
for i = 1:num
    target(i, y(i)) = 1;
end

% Transpose data
input = x';    
target = target';

% Before performing 10-fold validation we need to randomise the data
RandIndex = randperm(length(input));
input = input(:, RandIndex);
target = target(:, RandIndex);

% Create emotions network
trainingFcn = 'trainlm'; %traingd, traingda, traingdm, traingdx
learningFcn = 'learngdm'; %learngd or learngdm
NET = newff(input, target, 12, {'logsig','purelin'}, trainingFcn, learningFcn);

% Modify train parameters
NET.trainParam.show = 5;
NET.trainParam.epochs = 100;
NET.trainParam.goal = 1e-5;
NET.trainParam.lr = 0.01;
NET.trainParam.max_fail = 10; % validation check times (default is 6)   
NET.performParam.regularization = 0.005;   % Adds a regularization value which will penalize large weights and help avoid overfitting 

% Train network
[NET, TR] = train(NET, input, target);
save NET;
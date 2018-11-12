%kfoldresults = predicted ; actual
function f1score = f1score (predicted, actual)
    recall= 0;
    precision = 0;

    truepositive = 0;
    falsepositive = 0;
    falsenegative =0;

    for i=1:length(predicted)
        if predicted(i)==1 && actual(i) == 1
            truepositive = truepositive + 1;
        elseif predicted(i) == 1 && actual(i) == 0
            falsepositive = falsepositive + 1;
        elseif predicted(i) == 0 && actual(i) == 1
            falsenegative = falsenegative +1;
        end
    end
    
    recall = truepositive/(truepositive + falsenegative);
    precision = truepositive/(truepositive + falsepositive);
    
    f1score = 2*((precision * recall)/(precision + recall));
end
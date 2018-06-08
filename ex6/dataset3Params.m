function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%X(1:10,:)
%y(1:10)
%Xval(1:10,:)
%yval(1:10)

testValues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]


for c = 1:length(testValues)
    for s = 1:length(testValues)
    % SVMTRAIN(X, Y, C, kernelFunction, tol, max_passes) från help svmTrain
        trainedModel = svmTrain(X,y, testValues(c), @(x1, x2) gaussianKernel(x1, x2, testValues(s)));
        %pred = SVMPREDICT(model, X) från help svmPredict
        testPrediction = svmPredict(trainedModel, X);
        %mean(double(predictions ~= yval)) från ex6.pdf sid 9
        testResult(c,s) = mean(double(testPrediction ~= yval));
    end
end

%testResult för test av indexläsning. Data från träning av SVM ovan
%testResult=  [0.360000,   0.070000,   0.070000,  0.070000,   0.070000,   0.140000,   0.160000,   0.160000;
%   0.240000,   0.070000,   0.070000,   0.070000,   0.070000,   0.170000,   0.160000,   0.190000;
%   0.240000,   0.070000,   0.070000,   0.070000,   0.070000,   0.170000,   0.160000,   0.250000;
%   0.380000,   0.070000,   0.070000,   0.070000,   0.070000,   0.290000,   0.160000,   0.180000;
%   0.350000,   0.070000,   0.070000,   0.070000,   0.070000,   0.360000,   0.250000,   0.170000;
%   0.310000,   0.070000,   0.070000,   0.070000,   0.070000,   0.480000,   0.400000,   0.160000;
%   0.380000,   0.070000,   0.070000,   0.070000,   0.070000,   0.490000,   0.580000,   0.250000;
%   0.220000,   0.070000,   0.070000,   0.070000,   0.070000,   0.470000,   0.610000,   0.400000]

[minC, cIndex] = min(min(testResult,[],2));
[minSigma, sigmaIndex] = min(min(testResult,[],1));

C = testValues(cIndex);
sigma = testValues(sigmaIndex);

% =========================================================================

end

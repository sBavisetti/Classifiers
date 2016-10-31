close all;
clear all;
% Reading Training data from text files;
train_data = csvread('ATNT50_trainDataXY.txt');
test_data =  csvread('ATNT50_testDataXY.txt');

%transposing both training data and testing data  to put them in proper
%formats
train_data = train_data';
test_data = test_data';

%Knn Classifier:
knn_result = Knn(train_data,test_data,3)
knn_accuracy = Accuracy(knn_result)

% Centroid Classifier
centroid_result = centroid(train_data,test_data)
centroid_accuracy = Accuracy(centroid_result)

% Linear Regression Classifer
linear_regression_result = LinearRegression(train_data,test_data)
linear_regression_accuracy = Accuracy(linear_regression_result)
